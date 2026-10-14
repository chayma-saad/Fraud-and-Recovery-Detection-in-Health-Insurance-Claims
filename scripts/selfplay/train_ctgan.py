"""Trains a CTGAN generator on the real (raw, unencoded) claim feature table
and samples a synthetic claim pool for the RL environment to draw baseline
claim templates from — no real patient data reaches the RL sandbox past this
point (the environment only ever samples from `synthetic_claims.csv`).

Conditioned on RECOVERY_FLAG so the generator learns the (rare) recovered
claims' distribution too, not just the 99.93% majority class.
"""
import time

import pandas as pd
from ctgan import CTGAN

from common import (
    PROJECT_ROOT, RAW_COLS, CATEGORICAL_COLS, LABEL_COL,
    CTGAN_MODEL_PATH, SYNTHETIC_CLAIMS_PATH, load_raw_features,
)

CONSOLIDATED = PROJECT_ROOT / "data/processed/consolidated_claims_all_reports.csv"
ML_READY = PROJECT_ROOT / "data/processed/claims_ml_ready.csv"

TRAIN_SAMPLE_SIZE = 40_000  # subsample for tractable CTGAN training time
N_SYNTHETIC = 20_000
EPOCHS = 150


def main():
    df = load_raw_features(CONSOLIDATED, ML_READY)
    train = df[df["CLAIM_YEAR"] <= 2014][RAW_COLS + [LABEL_COL]]

    # Oversample the positive class into the training sample so CTGAN sees
    # enough recovered-claim examples to learn their shape, not just the
    # majority pattern (614 positives total is too few to trust a random
    # 40k subsample to include a representative number otherwise).
    pos = train[train[LABEL_COL] == 1]
    neg = train[train[LABEL_COL] == 0]
    neg_sample = neg.sample(n=min(TRAIN_SAMPLE_SIZE - len(pos), len(neg)), random_state=42)
    sample = pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1, random_state=42)
    print(f"CTGAN training sample: {len(sample):,} rows "
          f"({pos.shape[0]} positive, {neg_sample.shape[0]} negative)")

    discrete_columns = CATEGORICAL_COLS + [LABEL_COL]

    t0 = time.time()
    ctgan = CTGAN(epochs=EPOCHS, verbose=True, cuda=False)
    ctgan.fit(sample, discrete_columns)
    print(f"CTGAN training took {time.time() - t0:.0f}s")

    CTGAN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ctgan.save(str(CTGAN_MODEL_PATH))
    print(f"Saved: {CTGAN_MODEL_PATH}")

    synthetic = ctgan.sample(N_SYNTHETIC)
    # Physical sanity clip: CTGAN can propose small negative amounts for a
    # heavily right-skewed column; clip rather than resample; document the
    # clip rate as a data-quality check on the generator, not silently hide it.
    n_negative = (synthetic["TOTALAMOUNTCLAIMED"] < 0).sum()
    synthetic["TOTALAMOUNTCLAIMED"] = synthetic["TOTALAMOUNTCLAIMED"].clip(lower=1.0)
    synthetic["AGE"] = synthetic["AGE"].clip(lower=0, upper=100)
    print(f"Clipped {n_negative} ({n_negative/len(synthetic):.2%}) rows with negative claimed amount")

    SYNTHETIC_CLAIMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(SYNTHETIC_CLAIMS_PATH, index=False)
    print(f"Saved {len(synthetic):,} synthetic claims: {SYNTHETIC_CLAIMS_PATH}")
    print(f"Synthetic positive rate: {synthetic[LABEL_COL].mean():.4%} (real: {train[LABEL_COL].mean():.4%})")


if __name__ == "__main__":
    main()
