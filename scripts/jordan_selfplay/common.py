"""Shared feature schema, encoding, and paths for the Jordan-dataset
adversarial self-play pipeline -- the Idea 2 equivalent for the Jordan
data, mirroring scripts/selfplay/ but adapted to Jordan's schema and, since
jordan_ml_ready.csv is already a small (34-col), fully self-authored,
reproducible encoding (unlike Al Hilal's notebook-embedded 109-column
pipeline), simpler: the defender's feature set IS (most of) the Objective 1
feature set, minus the two claimedvalue-dependent peer features.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JORDAN_DIR = PROJECT_ROOT / "data/processed/jordan_data"
ML_READY_META_PATH = JORDAN_DIR / "jordan_ml_ready_meta.json"
ENCODED_SOURCE_PATH = JORDAN_DIR / "encoded_model_data.csv"

DEFENDER_V1_PATH = PROJECT_ROOT / "models/jordan_selfplay/defender_v1.json"
DEFENDER_V2_PATH = PROJECT_ROOT / "models/jordan_selfplay/defender_v2.json"
SYNTHETIC_CLAIMS_PATH = JORDAN_DIR / "selfplay/synthetic_claims.csv"
CTGAN_MODEL_PATH = PROJECT_ROOT / "models/jordan_selfplay/ctgan.pkl"
PPO_MODEL_PATH = PROJECT_ROOT / "models/jordan_selfplay/ppo_fraudster.zip"
TRAINING_LOG_PATH = JORDAN_DIR / "selfplay/ppo_training_log.csv"
ADVERSARIAL_EXAMPLES_PATH = JORDAN_DIR / "selfplay/adversarial_examples.csv"
RAW_CACHE_PATH = JORDAN_DIR / "selfplay/raw_features.csv"

LABEL_COL = "COLLUSION_FLAG"
LABEL_SRC = "is_collusion_adjacent_tmp"

CONTINUOUS_COLS = ["age_at_claim", "claimedvalue"]
CATEGORICAL_COLS = ["gender", "visitype", "dependency", "hcptype", "payment"]
# icd9code (2,023 distinct raw codes) is deliberately NOT one of CTGAN's
# discrete training columns -- that cardinality would blow up CTGAN's
# internal one-hot transformer for no real benefit here. Diagnosis code
# doesn't need to be *invented* by the generator; it's bootstrap-sampled
# from the real distribution instead, jointly with the other
# amount-independent columns below (see attach_sampled_extras), same
# device Al Hilal's Phase 4 used for timing/provider-history features.
RAW_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS

# Amount-independent columns, pulled pre-scaled from jordan_ml_ready.csv's
# own scaling (frozen per baseline claim -- the agent never controls
# timing/provider-history/dependency/diagnosis in this MVP, only the
# claimed amount).
EXTRA_SCALED_COLS = [
    "submission_lag_days", "patient_claim_count", "provider_claim_count",
    "patient_provider_interaction_count",
]
EXTRA_RAW_COLS = ["icd9code"]

AMOUNT_IDX = CONTINUOUS_COLS.index("claimedvalue")  # 1


def load_raw_features(cache=True):
    """Pulls the reduced raw (pre-encoding) feature set, row-aligned with
    jordan_ml_ready.csv's claim_year via the same source file both were
    built from (encoded_model_data.csv / model_features_curated.csv share
    row order -- both are direct exports of the same cleaned dataframe)."""
    if cache and RAW_CACHE_PATH.exists():
        return pd.read_csv(RAW_CACHE_PATH)

    cols = RAW_COLS + EXTRA_RAW_COLS + EXTRA_SCALED_COLS + ["claim_year", LABEL_SRC]
    df = pd.read_csv(ENCODED_SOURCE_PATH, usecols=cols, low_memory=False)
    df[LABEL_COL] = df[LABEL_SRC].astype(int)
    df = df.drop(columns=[LABEL_SRC])
    df = df.dropna(subset=["age_at_claim"]).reset_index(drop=True)

    if cache:
        RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CACHE_PATH, index=False)
    return df


def load_ml_ready_meta():
    with open(ML_READY_META_PATH) as f:
        return json.load(f)


def fit_encoder_meta(df):
    """Categorical level lists (fixed order) + continuous scaling params +
    ICD9 frequency map, fit on the real training data and frozen."""
    meta = {"categorical_levels": {}, "continuous_scale": {}}
    for col in CATEGORICAL_COLS:
        meta["categorical_levels"][col] = sorted(df[col].astype(str).unique().tolist())
    for col in CONTINUOUS_COLS:
        meta["continuous_scale"][col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}
    icd9 = df["icd9code"].astype(str)
    freq_map = icd9.value_counts(normalize=True)
    meta["icd9_freq_map"] = freq_map.to_dict()
    icd9_freq_vals = icd9.map(freq_map).fillna(0.0)
    meta["continuous_scale"]["ICD9_FREQ"] = {"mean": float(icd9_freq_vals.mean()), "std": float(icd9_freq_vals.std())}
    return meta


def save_encoder_meta(meta, path=None):
    path = path or (JORDAN_DIR / "selfplay/encoder_meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def load_encoder_meta(path=None):
    path = path or (JORDAN_DIR / "selfplay/encoder_meta.json")
    with open(path) as f:
        return json.load(f)


def encoded_feature_names(meta):
    names = [f"{col}_scaled" for col in CONTINUOUS_COLS] + ["ICD9_FREQ_scaled"]
    for col in CATEGORICAL_COLS:
        for level in meta["categorical_levels"][col]:
            names.append(f"{col}={level}")
    names += EXTRA_SCALED_COLS
    return names


LOG_AMOUNT_IDX = None  # Jordan has no log-amount column; only claimedvalue needs dynamic recompute


def encode_rows(df, meta):
    n = len(df)
    parts = []
    for col in CONTINUOUS_COLS:
        s = meta["continuous_scale"][col]
        parts.append(((df[col].values - s["mean"]) / s["std"]).reshape(-1, 1))

    icd9 = df["icd9code"].astype(str)
    freq_vals = icd9.map(meta["icd9_freq_map"]).fillna(0.0).values
    s = meta["continuous_scale"]["ICD9_FREQ"]
    parts.append(((freq_vals - s["mean"]) / s["std"]).reshape(-1, 1))

    for col in CATEGORICAL_COLS:
        levels = meta["categorical_levels"][col]
        level_index = {lvl: i for i, lvl in enumerate(levels)}
        onehot = np.zeros((n, len(levels)), dtype=np.float32)
        codes = df[col].astype(str).map(level_index)
        valid = codes.notna()
        onehot[np.where(valid)[0], codes[valid].astype(int).values] = 1.0
        parts.append(onehot)

    extra = df[EXTRA_SCALED_COLS].values.astype(np.float32)
    parts.append(extra)

    return np.hstack(parts).astype(np.float32)


def encode_amount_only(base_row_encoded, new_amount, meta, amount_idx=AMOUNT_IDX):
    """Recomputes only the claimedvalue entry for a new claimed amount --
    everything else (demographics, provider, diagnosis, timing) stays
    frozen at the baseline claim's values."""
    row = base_row_encoded.copy()
    s = meta["continuous_scale"]["claimedvalue"]
    row[amount_idx] = (new_amount - s["mean"]) / s["std"]
    return row


def attach_sampled_extras(df, real_df, rng):
    """For synthetic (CTGAN) claims, which have no real timing/provider
    history or diagnosis code: bootstrap-sample EXTRA_SCALED_COLS +
    EXTRA_RAW_COLS jointly (as whole rows, preserving their correlation)
    from the real dataset and attach them (documented approximation, same
    device as Al Hilal's Phase 4)."""
    cols = EXTRA_SCALED_COLS + EXTRA_RAW_COLS
    idx = rng.integers(0, len(real_df), size=len(df))
    sampled = real_df[cols].iloc[idx].reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    for col in cols:
        out[col] = sampled[col].values
    return out
