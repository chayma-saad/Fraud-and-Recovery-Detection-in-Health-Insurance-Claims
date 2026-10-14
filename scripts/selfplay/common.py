"""Shared feature schema, encoding, and paths for the Idea 2 (adversarial
self-play) pipeline. Kept deliberately separate from the Phase 2/3 pipeline:
this is a self-contained red-team simulator with its own reduced feature set,
not a reuse of claims_ml_ready.csv's 109-column encoding (see docs for why).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_COLS_CACHE = PROJECT_ROOT / "data/processed/selfplay/raw_features.csv"
ENCODER_META_PATH = PROJECT_ROOT / "models/selfplay/encoder_meta.json"
DEFENDER_V1_PATH = PROJECT_ROOT / "models/selfplay/defender_v1.json"
DEFENDER_V2_PATH = PROJECT_ROOT / "models/selfplay/defender_v2.json"
SYNTHETIC_CLAIMS_PATH = PROJECT_ROOT / "data/processed/selfplay/synthetic_claims.csv"
CTGAN_MODEL_PATH = PROJECT_ROOT / "models/selfplay/ctgan.pkl"
PPO_MODEL_PATH = PROJECT_ROOT / "models/selfplay/ppo_fraudster.zip"
TRAINING_LOG_PATH = PROJECT_ROOT / "data/processed/selfplay/ppo_training_log.csv"
ADVERSARIAL_EXAMPLES_PATH = PROJECT_ROOT / "data/processed/selfplay/adversarial_examples.csv"

# Pulled pre-scaled from claims_ml_ready.csv (row-aligned) rather than
# recomputed here: these are amount-independent (timing, provider/network
# history, resubmission), so the agent never needs them dynamically
# recomputed when it changes a claimed amount — they can be safely frozen
# per baseline claim and reused as-is from the Phase 2 pipeline's own scaling.
EXTRA_SCALED_COLS = [
    "MEMBER_TENURE_DAYS", "DAYS_TO_RECEPTION", "RESUBMISSION_FLAG",
    "PROVIDER_CLAIM_FREQ", "NETWORK_CLAIM_FREQ", "CLAIM_MONTH", "CLAIM_QUARTER",
    "CLAIM_IS_WEEKEND",
]

CONTINUOUS_COLS = ["AGE", "TOTALAMOUNTCLAIMED", "NETPREMIUM", "GROSSPREMIUM"]
CATEGORICAL_COLS = [
    "GENDER", "MARITAL STATUS", "AGEBAND", "DEPENDENCY", "NETWORK TYPE", "FOB",
    "CHRONIC", "PREEXISTING", "CLAIMTYPE", "PROVIDER EMIRATE", "PROVIDERTYPE",
    "ICD CHAPTER",
]
RAW_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS
LABEL_COL = "RECOVERY_FLAG"


def load_raw_features(consolidated_path, ml_ready_path, cache=True):
    """Pulls the reduced feature set from the raw processed export, fills the
    (real, documented) gaps the same way as a claims analyst would, and
    row-aligns CLAIM_YEAR + RECOVERY_FLAG from claims_ml_ready.csv (verified
    identical row order/count in the Phase 3 notebook)."""
    if cache and DATA_RAW_COLS_CACHE.exists():
        return pd.read_csv(DATA_RAW_COLS_CACHE)

    usecols = RAW_COLS + ["RECOVERY/EX-GRATIA"]
    df = pd.read_csv(consolidated_path, usecols=usecols, low_memory=False)
    aux = pd.read_csv(ml_ready_path, usecols=["CLAIM_YEAR"] + EXTRA_SCALED_COLS)
    assert len(df) == len(aux), "Row count mismatch between raw and ml-ready exports"
    for col in ["CLAIM_YEAR"] + EXTRA_SCALED_COLS:
        df[col] = aux[col].values
    df[LABEL_COL] = (df["RECOVERY/EX-GRATIA"] > 0).astype(int)
    df = df.drop(columns=["RECOVERY/EX-GRATIA"])

    df["PROVIDER EMIRATE"] = df["PROVIDER EMIRATE"].fillna("Unknown")
    df["NETPREMIUM"] = df["NETPREMIUM"].fillna(df["NETPREMIUM"].median())
    df["GROSSPREMIUM"] = df["GROSSPREMIUM"].fillna(df["GROSSPREMIUM"].median())

    if cache:
        DATA_RAW_COLS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_RAW_COLS_CACHE, index=False)
    return df


def fit_encoder_meta(df):
    """Freezes everything needed to encode a raw claim (real or synthetic)
    into the fixed-width numeric vector the defender/environment use:
    one-hot category lists (fixed order) + continuous scaling params, fit
    once on the real training data and reused everywhere downstream."""
    meta = {"categorical_levels": {}, "continuous_scale": {}}
    for col in CATEGORICAL_COLS:
        meta["categorical_levels"][col] = sorted(df[col].astype(str).unique().tolist())
    for col in CONTINUOUS_COLS:
        meta["continuous_scale"][col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}
    log_amt = np.log1p(df["TOTALAMOUNTCLAIMED"].values)
    meta["continuous_scale"]["CLAIM_AMOUNT_LOG"] = {"mean": float(log_amt.mean()), "std": float(log_amt.std())}
    return meta


def save_encoder_meta(meta, path=ENCODER_META_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def load_encoder_meta(path=ENCODER_META_PATH):
    with open(path) as f:
        return json.load(f)


def encoded_feature_names(meta):
    names = [f"{col}_scaled" for col in CONTINUOUS_COLS] + ["CLAIM_AMOUNT_LOG_scaled"]
    for col in CATEGORICAL_COLS:
        for level in meta["categorical_levels"][col]:
            names.append(f"{col}={level}")
    names += EXTRA_SCALED_COLS
    return names


def encode_rows(df, meta):
    """Vectorized encoder: raw claim rows -> fixed-width numeric matrix
    matching `encoded_feature_names`. Unknown categorical levels (can happen
    with CTGAN-sampled rows in principle) fall through to all-zero dummies
    for that column, same convention as a one-hot encoder's unknown handling."""
    n = len(df)
    parts = []
    for col in CONTINUOUS_COLS:
        s = meta["continuous_scale"][col]
        parts.append(((df[col].values - s["mean"]) / s["std"]).reshape(-1, 1))
    log_amt = np.log1p(df["TOTALAMOUNTCLAIMED"].values)
    s = meta["continuous_scale"]["CLAIM_AMOUNT_LOG"]
    parts.append(((log_amt - s["mean"]) / s["std"]).reshape(-1, 1))

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


def attach_sampled_extras(df, real_df, rng):
    """For synthetic (CTGAN) claims, which have no real submission timing or
    provider history: bootstrap-sample EXTRA_SCALED_COLS jointly (as whole
    rows, to preserve their correlation) from the real dataset and attach
    them. Documented approximation — see the self-play notebook/doc for the
    honest limitation this introduces."""
    idx = rng.integers(0, len(real_df), size=len(df))
    sampled = real_df[EXTRA_SCALED_COLS].iloc[idx].reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    for col in EXTRA_SCALED_COLS:
        out[col] = sampled[col].values
    return out


AMOUNT_IDX = CONTINUOUS_COLS.index("TOTALAMOUNTCLAIMED")  # 1
LOG_AMOUNT_IDX = len(CONTINUOUS_COLS)  # 4: CLAIM_AMOUNT_LOG is appended right after CONTINUOUS_COLS


def encode_amount_only(base_row_encoded, new_amount, meta, amount_idx=AMOUNT_IDX, log_idx=LOG_AMOUNT_IDX):
    """Given an already-encoded baseline row, returns a copy with only the
    TOTALAMOUNTCLAIMED and CLAIM_AMOUNT_LOG entries recomputed for a new
    claimed amount — everything else (demographics, provider, diagnosis)
    stays frozen at the baseline claim's values. This is what lets the RL
    environment cheaply re-score a claim after the agent changes the amount,
    without re-running the full encoder per step."""
    row = base_row_encoded.copy()
    s_amt = meta["continuous_scale"]["TOTALAMOUNTCLAIMED"]
    s_log = meta["continuous_scale"]["CLAIM_AMOUNT_LOG"]
    row[amount_idx] = (new_amount - s_amt["mean"]) / s_amt["std"]
    row[log_idx] = (np.log1p(new_amount) - s_log["mean"]) / s_log["std"]
    return row
