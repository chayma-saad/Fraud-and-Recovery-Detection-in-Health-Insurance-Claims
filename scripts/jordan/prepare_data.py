"""Builds data/processed/jordan_data/jordan_ml_ready.csv -- the Jordan
equivalent of Phase 2's claims_ml_ready.csv.

Input is model_features_curated.csv, which was already cleaned and partly
engineered outside this session (see preprocessing_log.csv,
data_quality_report.csv, feature_relevance_for_objectives.csv in the same
folder). This script's job is narrower and specific to this project's
modeling needs:

1. Define the label: COLLUSION_FLAG = is_collusion_adjacent_tmp (696 claims
   explicitly flagged by investigators as possible HCP/policy-holder
   collusion signals -- chosen over the alternative candidate,
   is_correction_R01 (112 rows, ambiguous "routine correction" semantics),
   because it's a direct, investigator-justified fraud-risk signal with
   more data to learn from, at a similar rarity to Al Hilal's 0.073%).
2. Exclude every column that leaks the adjudication OUTCOME -- the same
   discipline Phase 2 applied to Al Hilal's STATUS/TOTALPAYABLE/etc.
   Jordan's adjudication outputs are duevalue/rejected/coinsurance/debit/
   cost (the approved/rejected/copay amounts), processcode and everything
   derived from it (processcode_category_tmp, is_correction_R01,
   is_fb_status, processtatus), processdate and anything timed off it
   (processing_lag_days, total_cycle_days), and the four *note presence
   flags (which turn out to encode processtatus indirectly -- verified:
   they're 41.23% missing in exact lockstep with the fb/FULLY PROCESSED
   split). Only claimedvalue (what was originally claimed) and
   submission_lag_days (claim-to-reception gap, known at reception) survive
   from that group -- exactly mirroring Al Hilal's TOTALAMOUNTCLAIMED /
   DAYS_TO_RECEPTION treatment.
3. Frequency-encode icd9code (2,023 distinct raw codes -- free text
   description dropped, matching how Al Hilal's ICD CHAPTER was handled).
4. Scale continuous features; leave already-encoded 0/1 and one-hot columns
   as-is (mirrors Phase 2's scaling rule exactly).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JORDAN_DIR = PROJECT_ROOT / "data/processed/jordan_data"
IN_PATH = JORDAN_DIR / "model_features_curated.csv"
OUT_PATH = JORDAN_DIR / "jordan_ml_ready.csv"
META_PATH = JORDAN_DIR / "jordan_ml_ready_meta.json"

LABEL_SRC = "is_collusion_adjacent_tmp"
LABEL_COL = "COLLUSION_FLAG"

# Columns already one-hot/binary encoded upstream -- kept as-is.
ALREADY_ENCODED = [
    "gender_encoded", "visitype_encoded",
    "dependency_Brother-Sister", "dependency_CHILD", "dependency_HOF",
    "dependency_Other", "dependency_PARENT", "dependency_SPOUSE",
    "hcptype_Doctor", "hcptype_Emergency Center", "hcptype_In-Patient Facility",
    "hcptype_Lab", "hcptype_Medical Service Center", "hcptype_Medical Supplier",
    "hcptype_Pharmacy", "hcptype_Radiology Center",
    "payment_NETWORK", "payment_NON-NETWORK",
]

# Data-quality / timing flags that are genuinely known pre-adjudication.
SAFE_FLAGS = [
    "date_order_violation", "dob_before_1900_flag", "age_implausible_flag",
    "negative_lag_flag", "icd9code_is_missing",
]

CONTINUOUS_COLS = [
    "claimedvalue", "age_at_claim", "submission_lag_days",
    "patient_claim_count", "provider_claim_count", "patient_provider_interaction_count",
]

# Explicitly excluded, and why (all are post-adjudication outcomes, or
# derived from them, or identifiers/free text -- see module docstring):
EXCLUDED = [
    "duevalue", "rejected", "coinsurance", "debit", "cost",
    "processcode", "processcode_category_tmp", "is_correction_R01",
    "is_fb_status", "processtatus", "processdate",
    "processing_lag_days", "total_cycle_days",
    "has_claimedvaluenote", "has_duevaluenote", "has_rejectednote",
    "has_coinsurancenote", "has_debitnote", "cost_is_missing",
    "nid", "hofid", "hcpid", "policyid", "claimid",
    "gender", "visitype", "dependency", "hcptype", "payment",
    "dateclaim", "datereceived", "dob", "effectivedate", "visitdate",
    "icd9code", "icd9codedescrip", "claimeditem", "source_file",
]


def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Loaded {IN_PATH.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df[LABEL_COL] = df[LABEL_SRC].astype(int)
    print(f"Label = {LABEL_SRC}: {df[LABEL_COL].sum()} positive ({df[LABEL_COL].mean():.4%})")

    # Frequency-encode diagnosis code.
    icd9 = df["icd9code"].astype(str)
    freq_map = icd9.value_counts(normalize=True)
    df["ICD9_FREQ"] = icd9.map(freq_map).fillna(0.0)

    # Peer-deviation baseline feature, mirroring Phase 2's PEER_ZSCORE /
    # AMOUNT_OVER_MEDIAN: group by (3-digit ICD9 category, dependency,
    # visit type) and measure how far this claim's amount sits from its
    # peer group's typical amount.
    icd9_chapter_proxy = icd9.str.slice(0, 3)
    peer_group = icd9_chapter_proxy.astype(str) + "|" + df["dependency"].astype(str) + "|" + df["visitype"].astype(str)
    peer_stats = df.groupby(peer_group)["claimedvalue"].agg(["mean", "std", "median"])
    peer_stats["std"] = peer_stats["std"].replace(0, np.nan)
    joined = peer_group.map(peer_stats["mean"]).rename("PEER_MEAN")
    peer_std = peer_group.map(peer_stats["std"])
    peer_median = peer_group.map(peer_stats["median"]).replace(0, np.nan)
    df["PEER_ZSCORE"] = ((df["claimedvalue"] - joined) / peer_std).fillna(0.0)
    df["AMOUNT_OVER_MEDIAN"] = (df["claimedvalue"] / peer_median).fillna(1.0)

    keep_cols = (CONTINUOUS_COLS + ["ICD9_FREQ", "PEER_ZSCORE", "AMOUNT_OVER_MEDIAN"]
                 + ALREADY_ENCODED + SAFE_FLAGS)
    missing = [c for c in keep_cols if c not in df.columns]
    assert not missing, f"Expected columns missing: {missing}"

    feature_df = df[keep_cols].copy()
    feature_df["claim_year"] = df["claim_year"]
    feature_df[LABEL_COL] = df[LABEL_COL]

    n_missing_age = feature_df["age_at_claim"].isna().sum()
    if n_missing_age:
        print(f"Dropping {n_missing_age} rows ({n_missing_age/len(feature_df):.4%}) "
              f"missing age_at_claim (invalid/missing dob) -- negligible, not imputed")
        feature_df = feature_df.dropna(subset=["age_at_claim"]).reset_index(drop=True)

    # Scale continuous features (fit on the whole dataset, same convention
    # Phase 2 used -- CLAIM_YEAR itself stays raw for the temporal split).
    scale_cols = CONTINUOUS_COLS + ["ICD9_FREQ", "PEER_ZSCORE", "AMOUNT_OVER_MEDIAN"]
    scaler = StandardScaler()
    feature_df[scale_cols] = scaler.fit_transform(feature_df[scale_cols])

    n_before = len(feature_df)
    n_na = feature_df.isna().sum().sum()
    print(f"NaNs remaining after feature selection: {n_na}")
    assert n_na == 0, "Unexpected NaNs in final feature set"

    feature_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH.name}: {feature_df.shape[0]:,} rows x {feature_df.shape[1]} columns")

    meta = {
        "label_source_column": LABEL_SRC,
        "label_column": LABEL_COL,
        "n_positive": int(df[LABEL_COL].sum()),
        "n_total": n_before,
        "excluded_columns": EXCLUDED,
        "continuous_scaled": scale_cols,
        "scaler_mean": {c: float(m) for c, m in zip(scale_cols, scaler.mean_)},
        "scaler_scale": {c: float(s) for c, s in zip(scale_cols, scaler.scale_)},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {META_PATH.name}")


if __name__ == "__main__":
    main()
