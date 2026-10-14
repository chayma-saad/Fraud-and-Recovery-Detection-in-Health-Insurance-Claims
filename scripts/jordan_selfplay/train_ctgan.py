"""Trains a CTGAN generator on real Jordan claim features and samples a
synthetic pool for the RL environment -- mirrors scripts/selfplay/train_ctgan.py."""
import time

import pandas as pd
from ctgan import CTGAN

from common import (
    RAW_COLS, CATEGORICAL_COLS, LABEL_COL, CTGAN_MODEL_PATH,
    SYNTHETIC_CLAIMS_PATH, load_raw_features,
)

TRAIN_SAMPLE_SIZE = 40_000
N_SYNTHETIC = 20_000
EPOCHS = 150


def main():
    df = load_raw_features()
    train = df[df["claim_year"] <= 2015][RAW_COLS + [LABEL_COL]]

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
    n_negative = (synthetic["claimedvalue"] < 0).sum()
    synthetic["claimedvalue"] = synthetic["claimedvalue"].clip(lower=0.1)
    synthetic["age_at_claim"] = synthetic["age_at_claim"].clip(lower=0, upper=100)
    print(f"Clipped {n_negative} ({n_negative/len(synthetic):.2%}) rows with negative claimed amount")

    SYNTHETIC_CLAIMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(SYNTHETIC_CLAIMS_PATH, index=False)
    print(f"Saved {len(synthetic):,} synthetic claims: {SYNTHETIC_CLAIMS_PATH}")
    print(f"Synthetic positive rate: {synthetic[LABEL_COL].mean():.4%} (real: {train[LABEL_COL].mean():.4%})")


if __name__ == "__main__":
    main()
