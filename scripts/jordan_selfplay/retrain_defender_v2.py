"""One round of defender retraining -- Jordan-dataset equivalent of
scripts/selfplay/retrain_defender_v2.py."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from stable_baselines3 import PPO

from common import (
    PROJECT_ROOT, LABEL_COL, load_encoder_meta, encode_rows, encode_amount_only,
    AMOUNT_IDX, load_raw_features, attach_sampled_extras,
    DEFENDER_V2_PATH, SYNTHETIC_CLAIMS_PATH, PPO_MODEL_PATH, ADVERSARIAL_EXAMPLES_PATH,
)
from fraud_env import ClaimFraudEnv
from train_ppo import evaluate_policy_attack_success, EVAL_EPISODES


def main():
    meta = load_encoder_meta()
    real_df = load_raw_features()
    train = real_df[real_df["claim_year"] <= 2015]
    holdout = real_df[real_df["claim_year"] > 2015]

    X_train_real = encode_rows(train, meta)
    y_train_real = train[LABEL_COL].values

    synthetic = pd.read_csv(SYNTHETIC_CLAIMS_PATH)
    rng = np.random.default_rng(7)
    synthetic = attach_sampled_extras(synthetic, train, rng)
    synthetic_encoded_pool = encode_rows(synthetic, meta)

    adv = pd.read_csv(ADVERSARIAL_EXAMPLES_PATH)
    print(f"Adversarial examples discovered by the trained agent: {len(adv):,} evasive sub-claims")

    adv_rows = np.stack([
        encode_amount_only(synthetic_encoded_pool[int(r.claim_idx)], r.sub_claim_amount, meta,
                            amount_idx=AMOUNT_IDX)
        for r in adv.itertuples()
    ])
    adv_labels = np.ones(len(adv_rows))

    X_train_v2 = np.vstack([X_train_real, adv_rows])
    y_train_v2 = np.concatenate([y_train_real, adv_labels])
    print(f"Defender v2 training set: {len(y_train_v2):,} rows "
          f"({y_train_real.sum():,.0f} original positives + {len(adv_rows):,} adversarial positives)")

    pos = y_train_v2.sum()
    neg = len(y_train_v2) - pos
    spw = neg / pos

    defender_v2 = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        scale_pos_weight=spw, random_state=42, eval_metric="aucpr", tree_method="hist",
    )
    defender_v2.fit(X_train_v2, y_train_v2)

    X_holdout = encode_rows(holdout, meta)
    y_holdout = holdout[LABEL_COL].values
    proba = defender_v2.predict_proba(X_holdout)[:, 1]
    prauc = average_precision_score(y_holdout, proba)
    rocauc = roc_auc_score(y_holdout, proba)
    print(f"Defender v2 holdout PR-AUC: {prauc:.4f}, ROC-AUC: {rocauc:.4f}")

    DEFENDER_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    defender_v2.save_model(str(DEFENDER_V2_PATH))

    train_proba_v2 = defender_v2.predict_proba(X_train_v2)[:, 1]
    threshold_v2 = float(np.quantile(train_proba_v2, 0.95))
    print(f"Defender v2 detection threshold: {threshold_v2:.4f}")

    model = PPO.load(str(PPO_MODEL_PATH))
    eval_env_v2 = ClaimFraudEnv(synthetic, meta, defender_v2, threshold_v2, seed=3)
    success_v2, kept_v2, _ = evaluate_policy_attack_success(eval_env_v2, model, EVAL_EPISODES, rng_seed=55_555)
    print(f"\nSame trained agent vs. defender_v2: {success_v2:.1%} attack success, {kept_v2:.1%} mean budget kept")

    with open(PROJECT_ROOT / "models/jordan_selfplay/defender_v2_meta.json", "w") as f:
        json.dump({
            "threshold": threshold_v2, "holdout_prauc": prauc, "holdout_rocauc": rocauc,
            "n_adversarial_examples_added": len(adv_rows),
            "frozen_agent_attack_success_rate_vs_v2": success_v2,
            "frozen_agent_mean_kept_fraction_vs_v2": kept_v2,
        }, f, indent=2)


if __name__ == "__main__":
    main()
