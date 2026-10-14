"""Builds data/processed/app_data.json: everything the interactive
'Claims Lab' HTML app needs, precomputed once in Python so the app itself
is a static file with no server, no account, and no live model dependency
other than a small hand-written JS XGBoost tree evaluator (verified
bit-for-bit against xgboost's own predict_proba -- see chat history).

Three payloads:
  1. defenders   -- slimmed XGBoost tree arrays for defender_v1 and
                    defender_v2, small enough to embed inline, exact enough
                    to reproduce predict_proba() in pure JS.
  2. simulator    -- ~30 curated synthetic baseline claims (human-readable
                     summary + precomputed 81-dim encoded vector) for the
                     "Beat the Defender" manual mode, plus ~20 full episode
                     replay traces (the ACTUAL trained PPO agent's recorded
                     actions) for the "Watch the Agent" mode, replayed
                     against both defender_v1 and defender_v2.
  3. shap         -- ~24 curated real 2016 test claims with predicted
                     probability, actual label, SHAP base value, and the
                     top contributing features (name, raw value, SHAP
                     value) from Phase 3's XGBoost model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "selfplay"))

from xgboost import XGBClassifier
from stable_baselines3 import PPO
import shap

from common import (
    load_encoder_meta, load_raw_features, encode_rows, encode_amount_only,
    attach_sampled_extras, AMOUNT_IDX, LOG_AMOUNT_IDX, CONTINUOUS_COLS,
    CATEGORICAL_COLS, EXTRA_SCALED_COLS, LABEL_COL, encoded_feature_names,
)
from fraud_env import ClaimFraudEnv, FRAUD_BUDGET_MULTIPLIER, MAX_STEPS

OUT_PATH = PROJECT_ROOT / "data/processed/app_data.json"


# ---------------------------------------------------------------------------
def slim_tree(t):
    return {
        "l": t["left_children"], "r": t["right_children"],
        "f": t["split_indices"], "c": t["split_conditions"],
    }


def load_defender_trees(path):
    d = json.load(open(path))
    gb = d["learner"]["gradient_booster"]["model"]
    base_score = float(d["learner"]["learner_model_param"]["base_score"].strip("[]"))
    return {"trees": [slim_tree(t) for t in gb["trees"]], "baseScore": base_score}


# ---------------------------------------------------------------------------
def build_simulator_payload(meta):
    synthetic = pd.read_csv(PROJECT_ROOT / "data/processed/selfplay/synthetic_claims.csv")
    real_df = load_raw_features(
        PROJECT_ROOT / "data/processed/consolidated_claims_all_reports.csv",
        PROJECT_ROOT / "data/processed/claims_ml_ready.csv",
    )
    real_train = real_df[real_df["CLAIM_YEAR"] <= 2014]
    rng = np.random.default_rng(42)
    synthetic = attach_sampled_extras(synthetic, real_train, rng)
    encoded_pool = encode_rows(synthetic, meta)

    # --- Curated baseline claims for manual "Beat the Defender" mode ---
    picks = synthetic.sample(n=30, random_state=11).index.tolist()
    baseline_claims = []
    for idx in picks:
        row = synthetic.loc[idx]
        baseline_claims.append({
            "id": int(idx),
            "amount": round(float(row["TOTALAMOUNTCLAIMED"]), 2),
            "age": round(float(row["AGE"]), 0),
            "ageband": row["AGEBAND"],
            "gender": row["GENDER"],
            "provider_type": row["PROVIDERTYPE"],
            "provider_emirate": row["PROVIDER EMIRATE"],
            "network_type": row["NETWORK TYPE"],
            "fob": row["FOB"],
            "icd_chapter": row["ICD CHAPTER"],
            "chronic": row["CHRONIC"],
            "encoded": [round(float(v), 5) for v in encoded_pool[idx].tolist()],
        })

    # --- Full episode replay traces from the ACTUAL trained agent ---
    defender_v1 = XGBClassifier()
    defender_v1.load_model(str(PROJECT_ROOT / "models/selfplay/defender_v1.json"))
    defender_v2 = XGBClassifier()
    defender_v2.load_model(str(PROJECT_ROOT / "models/selfplay/defender_v2.json"))
    with open(PROJECT_ROOT / "models/selfplay/defender_v1_meta.json") as f:
        threshold_v1 = json.load(f)["threshold"]
    with open(PROJECT_ROOT / "models/selfplay/defender_v2_meta.json") as f:
        threshold_v2 = json.load(f)["threshold"]

    model = PPO.load(str(PROJECT_ROOT / "models/selfplay/ppo_fraudster.zip"))
    env = ClaimFraudEnv(synthetic, meta, defender_v1, threshold_v1, seed=123)

    # First pass: collect a large pool of clean (no degenerate near-zero
    # submissions) candidate episodes, tagged with their step count, so
    # multi-step ("splitting") episodes -- genuinely rare, ~10% of
    # successful episodes per the Step 6 finding -- can be deliberately
    # over-represented in the curated set rather than crowded out by the
    # far more common single-shot ones.
    candidates = []
    ep_seed = 0
    while len(candidates) < 200 and ep_seed < 3000:
        obs, _ = env.reset(seed=1000 + ep_seed)
        ep_seed += 1
        claim_idx = env._claim_idx
        budget_total = env._budget_total

        steps_v1 = []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_v1.append({
                "amount": round(float(info["claimed_amount"]), 2),
                "proba": round(float(info["detect_proba"]), 5),
                "caught": bool(info["caught"]),
                "kept_fraction": round(float(info["kept_fraction"]), 4),
            })
        if any(s["amount"] < 0.01 * budget_total for s in steps_v1):
            continue
        candidates.append({
            "claim_idx": claim_idx, "budget_total": budget_total, "steps_v1": steps_v1,
            "outcome_v1": "caught" if steps_v1[-1]["caught"] else "success",
        })
    print(f"Collected {len(candidates)} clean candidate episodes from {ep_seed} rollouts")

    multi = [c for c in candidates if len(c["steps_v1"]) >= 2]
    single = [c for c in candidates if len(c["steps_v1"]) == 1]
    print(f"  of which {len(multi)} are multi-step (split)")

    chosen = multi[:8]
    seen_buckets = {"caught_v1": sum(1 for c in chosen if c["outcome_v1"] == "caught"),
                     "success_v1": sum(1 for c in chosen if c["outcome_v1"] == "success")}
    for c in single:
        if len(chosen) >= 24:
            break
        key = f"{c['outcome_v1']}_v1"
        if seen_buckets.get(key, 0) >= 14:
            continue
        seen_buckets[key] = seen_buckets.get(key, 0) + 1
        chosen.append(c)

    episodes = []
    for c in chosen:
        claim_idx = c["claim_idx"]
        budget_total = c["budget_total"]
        steps_v1 = c["steps_v1"]
        outcome_v1 = c["outcome_v1"]
        base_row = synthetic.loc[claim_idx]
        base_amount = base_row["TOTALAMOUNTCLAIMED"]

        # Replay the exact same claimed amounts against defender_v2.
        base_encoded = encoded_pool[claim_idx]
        steps_v2 = []
        for s in steps_v1:
            row = encode_amount_only(base_encoded, s["amount"], meta,
                                      amount_idx=AMOUNT_IDX, log_idx=LOG_AMOUNT_IDX)
            p2 = float(defender_v2.predict_proba(row.reshape(1, -1))[0, 1])
            caught2 = p2 >= threshold_v2
            steps_v2.append({"amount": s["amount"], "proba": round(p2, 5), "caught": bool(caught2)})
            if caught2:
                break

        episodes.append({
            "claim": {
                "id": int(claim_idx),
                "amount": round(float(base_amount), 2),
                "ageband": base_row["AGEBAND"], "gender": base_row["GENDER"],
                "provider_type": base_row["PROVIDERTYPE"], "provider_emirate": base_row["PROVIDER EMIRATE"],
                "network_type": base_row["NETWORK TYPE"], "fob": base_row["FOB"],
                "icd_chapter": base_row["ICD CHAPTER"],
            },
            "budget_total": round(float(budget_total), 2),
            "steps_v1": steps_v1,
            "steps_v2": steps_v2,
            "outcome_v1": outcome_v1,
            "outcome_v2": "caught" if (steps_v2 and steps_v2[-1]["caught"]) else "success",
        })

    print(f"Built {len(baseline_claims)} baseline claims, {len(episodes)} episode replays")
    return {
        "budgetMultiplier": FRAUD_BUDGET_MULTIPLIER,
        "maxSteps": MAX_STEPS,
        "thresholdV1": threshold_v1,
        "thresholdV2": threshold_v2,
        "baselineClaims": baseline_claims,
        "episodes": episodes,
        "amountIdx": AMOUNT_IDX,
        "logAmountIdx": LOG_AMOUNT_IDX,
        "continuousScale": {
            "amount": meta["continuous_scale"]["TOTALAMOUNTCLAIMED"],
            "log": meta["continuous_scale"]["CLAIM_AMOUNT_LOG"],
        },
    }


# ---------------------------------------------------------------------------
PHASE3_FEATURES_DISPLAY = {
    "PEER_ZSCORE": "Peer-group deviation (z-score)",
    "AMOUNT_OVER_MEDIAN": "Amount vs. peer median",
    "PEER_MEDIAN": "Peer group's median amount",
    "PEER_MEAN": "Peer group's mean amount",
    "PEER_STD": "Peer group's amount spread",
    "PROVIDER_AVG_AMOUNT": "Provider's average claim amount",
    "PROVIDER_MEDIAN_AMOUNT": "Provider's median claim amount",
    "PROVIDER_STD_AMOUNT": "Provider's amount spread",
    "CLAIM_AMOUNT_LOG": "Claim amount (log)",
    "MEMBER_TENURE_DAYS": "Member tenure (days)",
    "DAYS_TO_RECEPTION": "Days to claim reception",
    "POLICY_DAYS_REMAINING": "Days until policy expiry",
    "PROVIDER_CLAIM_FREQ": "Provider claim frequency",
    "NETWORK_CLAIM_FREQ": "Network claim frequency",
    "AGE": "Beneficiary age",
    "TOTALAMOUNTCLAIMED": "Total amount claimed",
    "BENEFICIARYAMOUNTCLAIMED": "Beneficiary-paid share",
    "PAYERAMOUNTCLAIMED": "Payer-paid share",
    "NETPREMIUM": "Policy net premium",
    "GROSSPREMIUM": "Policy gross premium",
    "ICD CHAPTER_FREQ": "Diagnosis chapter frequency",
    "ICD SUB CHAPTER_FREQ": "Diagnosis sub-chapter frequency",
    "PRODUCT_FREQ": "Product frequency",
    "NATIONALITY_FREQ": "Nationality frequency",
    "EMPNO_FREQ": "Employer frequency",
    "CATEGORY_FREQ": "Policy category frequency",
    "NETWORK_FREQ": "Network frequency",
    "PROVIDERCOUNTRY_FREQ": "Provider country frequency",
    "PROVIDER GROUP_FREQ": "Provider group frequency",
    "PROVIDER_FREQ": "Provider frequency",
    "RESUBMISSION_FLAG": "Was resubmitted",
    "CLAIM_MONTH": "Claim month",
    "CLAIM_QUARTER": "Claim quarter",
    "CLAIM_IS_WEEKEND": "Submitted on a weekend",
    "REPORTING AUTHORITY_DHA": "Reporting Authority: DHA",
    "REPORTING AUTHORITY_HAAD": "Reporting Authority: HAAD",
    "REPORTING AUTHORITY_Unknown": "Reporting Authority: Unknown",
}


def build_shap_payload():
    ml_ready = pd.read_csv(PROJECT_ROOT / "data/processed/claims_ml_ready.csv")
    features = [c for c in ml_ready.columns if c != "RECOVERY_FLAG"]
    train = ml_ready[ml_ready["CLAIM_YEAR"] <= 2014]
    test = ml_ready[ml_ready["CLAIM_YEAR"] == 2016].reset_index(drop=True)

    pos = train["RECOVERY_FLAG"].sum()
    spw = (len(train) - pos) / pos
    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        scale_pos_weight=spw, random_state=42, eval_metric="aucpr", tree_method="hist",
    )
    model.fit(train[features], train["RECOVERY_FLAG"])

    X_test = test[features]
    proba = model.predict_proba(X_test)[:, 1]
    test = test.assign(P_RECOVERY=proba)

    explainer = shap.TreeExplainer(model)

    recovered = test[test["RECOVERY_FLAG"] == 1].sort_values("P_RECOVERY", ascending=False)
    top_missed = test[test["RECOVERY_FLAG"] == 0].sort_values("P_RECOVERY", ascending=False).head(8)
    low_risk = test[test["RECOVERY_FLAG"] == 0].sort_values("P_RECOVERY").head(6)
    picks = pd.concat([recovered.head(10), top_missed, low_risk]).drop_duplicates()
    picks = picks.head(24)

    raw = pd.read_csv(
        PROJECT_ROOT / "data/processed/consolidated_claims_all_reports.csv",
        usecols=["TOTALAMOUNTCLAIMED", "PROVIDERTYPE", "PROVIDER EMIRATE", "FOB",
                 "AGEBAND", "ICD CHAPTER", "AGE", "GENDER"],
        low_memory=False,
    )
    test_reset = test.reset_index(drop=True)
    orig_index_map = ml_ready[ml_ready["CLAIM_YEAR"] == 2016].index.values

    claims_out = []
    for pos_in_test, row in picks.iterrows():
        orig_idx = orig_index_map[pos_in_test]
        raw_row = raw.loc[orig_idx]
        x = X_test.loc[[pos_in_test]]
        sv = explainer.shap_values(x)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            arr = np.atleast_1d(base_value)
            base_value = float(arr[1]) if len(arr) > 1 else float(arr[0])

        order = np.argsort(-np.abs(sv))[:10]
        contributions = []
        for i in order:
            fname = features[i]
            display = PHASE3_FEATURES_DISPLAY.get(fname)
            if display is None:
                display = fname
                for col in CATEGORICAL_COLS:
                    prefix = col + "_"
                    if fname.startswith(prefix):
                        display = f"{col.title()}: {fname[len(prefix):]}"
                        break
            contributions.append({
                "feature": display, "raw_feature": fname,
                "shap": round(float(sv[i]), 4),
            })

        claims_out.append({
            "id": int(orig_idx),
            "amount": round(float(raw_row["TOTALAMOUNTCLAIMED"]), 2),
            "provider_type": str(raw_row["PROVIDERTYPE"]),
            "provider_emirate": str(raw_row["PROVIDER EMIRATE"]) if pd.notna(raw_row["PROVIDER EMIRATE"]) else "Unknown",
            "fob": str(raw_row["FOB"]),
            "ageband": str(raw_row["AGEBAND"]),
            "icd_chapter": str(raw_row["ICD CHAPTER"])[:60],
            "actual_label": int(test_reset.loc[pos_in_test, "RECOVERY_FLAG"]),
            "predicted_proba": round(float(test_reset.loc[pos_in_test, "P_RECOVERY"]), 5),
            "base_value": round(float(base_value), 5),
            "contributions": contributions,
        })

    print(f"Built SHAP payload: {len(claims_out)} claims")
    return {"claims": claims_out}


# ---------------------------------------------------------------------------
def main():
    meta = load_encoder_meta()

    defenders = {
        "v1": load_defender_trees(PROJECT_ROOT / "models/selfplay/defender_v1.json"),
        "v2": load_defender_trees(PROJECT_ROOT / "models/selfplay/defender_v2.json"),
    }
    simulator = build_simulator_payload(meta)
    shap_payload = build_shap_payload()

    payload = {"defenders": defenders, "simulator": simulator, "shap": shap_payload}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
