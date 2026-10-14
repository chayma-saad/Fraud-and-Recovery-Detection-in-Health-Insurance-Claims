"""Trains the static XGBoost defender (v1) the fraudster agent will attack.

Deliberately a fresh, smaller-feature-set classifier rather than a reload of
the Phase 3 XGBoost model: the self-play environment needs to cheaply
re-encode a claim every time the agent changes its amount (see
common.encode_amount_only), which requires knowing exactly how every input
feature was built. Phase 3's 109-column pipeline (frequency encoding fit
dataset-wide, peer-group stats, provider aggregates) isn't reproducible at
that granularity outside the notebook it was built in. This defender uses a
smaller, fully self-contained feature set (18 raw columns -> ~78 encoded
dims) built and encoded entirely within this module, so every step of the
simulator is exactly reproducible.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from common import (
    PROJECT_ROOT, LABEL_COL, DEFENDER_V1_PATH, ENCODER_META_PATH,
    load_raw_features, fit_encoder_meta, save_encoder_meta, encode_rows,
    encoded_feature_names,
)

CONSOLIDATED = PROJECT_ROOT / "data/processed/consolidated_claims_all_reports.csv"
ML_READY = PROJECT_ROOT / "data/processed/claims_ml_ready.csv"


def main():
    df = load_raw_features(CONSOLIDATED, ML_READY)
    print(f"Raw feature table: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Positive class: {df[LABEL_COL].sum()} ({df[LABEL_COL].mean():.4%})")

    train = df[df["CLAIM_YEAR"] <= 2014]
    holdout = df[df["CLAIM_YEAR"] > 2014]

    meta = fit_encoder_meta(train)
    save_encoder_meta(meta)
    feat_names = encoded_feature_names(meta)
    print(f"Encoded feature width: {len(feat_names)}")

    X_train = encode_rows(train, meta)
    y_train = train[LABEL_COL].values
    X_holdout = encode_rows(holdout, meta)
    y_holdout = holdout[LABEL_COL].values

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos

    defender = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        scale_pos_weight=spw, random_state=42, eval_metric="aucpr", tree_method="hist",
    )
    defender.fit(X_train, y_train)

    proba = defender.predict_proba(X_holdout)[:, 1]
    prauc = average_precision_score(y_holdout, proba)
    rocauc = roc_auc_score(y_holdout, proba)
    print(f"Defender v1 holdout (2015-2016) PR-AUC: {prauc:.4f}, ROC-AUC: {rocauc:.4f}")

    order = np.argsort(-proba)
    k = int(np.ceil(len(y_holdout) * 0.05))
    recall5 = y_holdout[order[:k]].sum() / y_holdout.sum()
    print(f"Recall@5% on holdout: {recall5:.1%}")

    DEFENDER_V1_PATH.parent.mkdir(parents=True, exist_ok=True)
    defender.save_model(str(DEFENDER_V1_PATH))
    print(f"Saved: {DEFENDER_V1_PATH}")

    # A fixed detection threshold the environment uses to call a claim
    # "caught": the probability cutoff that would flag the top 5% riskiest
    # training claims, mirroring the Phase 3 audit-queue framing.
    train_proba = defender.predict_proba(X_train)[:, 1]
    threshold = float(np.quantile(train_proba, 0.95))
    print(f"Detection threshold (95th pct of train scores): {threshold:.4f}")

    with open(PROJECT_ROOT / "models/selfplay/defender_v1_meta.json", "w") as f:
        json.dump({
            "threshold": threshold, "holdout_prauc": prauc, "holdout_rocauc": rocauc,
            "holdout_recall_at_5pct": float(recall5), "feature_names": feat_names,
        }, f, indent=2)


if __name__ == "__main__":
    main()
