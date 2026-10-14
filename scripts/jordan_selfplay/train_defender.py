"""Trains the static XGBoost defender (v1) the fraudster agent will attack
-- Jordan-dataset equivalent of scripts/selfplay/train_defender.py."""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from common import (
    PROJECT_ROOT, LABEL_COL, DEFENDER_V1_PATH,
    load_raw_features, fit_encoder_meta, save_encoder_meta, encode_rows,
    encoded_feature_names,
)


def main():
    df = load_raw_features()
    print(f"Raw feature table: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Positive class: {df[LABEL_COL].sum()} ({df[LABEL_COL].mean():.4%})")

    train = df[df["claim_year"] <= 2015]
    holdout = df[df["claim_year"] > 2015]

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
    print(f"Defender v1 holdout (2016-2017) PR-AUC: {prauc:.4f}, ROC-AUC: {rocauc:.4f}")

    order = np.argsort(-proba)
    k = int(np.ceil(len(y_holdout) * 0.05))
    recall5 = y_holdout[order[:k]].sum() / y_holdout.sum()
    print(f"Recall@5% on holdout: {recall5:.1%}")

    DEFENDER_V1_PATH.parent.mkdir(parents=True, exist_ok=True)
    defender.save_model(str(DEFENDER_V1_PATH))
    print(f"Saved: {DEFENDER_V1_PATH}")

    train_proba = defender.predict_proba(X_train)[:, 1]
    threshold = float(np.quantile(train_proba, 0.95))
    print(f"Detection threshold (95th pct of train scores): {threshold:.4f}")

    with open(PROJECT_ROOT / "models/jordan_selfplay/defender_v1_meta.json", "w") as f:
        json.dump({
            "threshold": threshold, "holdout_prauc": prauc, "holdout_rocauc": rocauc,
            "holdout_recall_at_5pct": float(recall5), "feature_names": feat_names,
        }, f, indent=2)


if __name__ == "__main__":
    main()
