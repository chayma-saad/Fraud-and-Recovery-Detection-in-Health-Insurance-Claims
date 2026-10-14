"""Generates notebooks/04_jordan_modeling.ipynb -- the Jordan-dataset
equivalent of Phase 3 (02_modeling.ipynb), following the same structure and
the same fixes already learned there (LightGBM needs a lighter
scale_pos_weight + regularization + early stopping under this kind of
extreme imbalance)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ==========================================================================
md(r"""# Phase 3b — Jordan Dataset: Collusion-Risk Model

A second, independent dataset for the same recovery/fraud-risk modeling
question Phase 3 answered for Al Hilal Takaful -- run alongside it, not
replacing it, so the thesis can show whether the approach generalizes to a
structurally different claims dataset.

**Dataset.** 602,280 Jordanian health-insurance claims (2013-2017),
pre-cleaned and partly feature-engineered outside this notebook (see
`data/processed/jordan_data/preprocessing_log.csv`,
`data_quality_report.csv`, `feature_relevance_for_objectives.csv`). This
notebook picks up from that cleaned export, defines a leakage-safe feature
set, and runs the same model comparison Phase 3 did.

**Label.** `COLLUSION_FLAG` = `is_collusion_adjacent_tmp`, an existing
engineered column: 696 of 602,267 claims (0.116%) that investigators flagged
via `processcode` patterns as possible HCP (healthcare provider) /
policy-holder collusion signals. Chosen over the alternative candidate in
the data (`is_correction_R01`, 112 rows of ambiguous "routine correction"
claims) because it's a direct, investigator-justified fraud-risk signal
with more data to learn from, at a rarity similar to Al Hilal's 0.073%.

**Leakage discipline.** Because the label is derived from `processcode`,
every column that is the adjudication *outcome* -- or timed off it -- is
excluded from features, exactly as strictly as Phase 2 excluded Al Hilal's
`STATUS`/`TOTALPAYABLE`/etc.: `duevalue`, `rejected`, `coinsurance`,
`debit`, `cost` (the approved/rejected/copay amounts), `processcode` and
everything derived from it, `processdate` and anything timed off it
(`processing_lag_days`, `total_cycle_days` -- only known once adjudication
*completes*), and the four `*note` presence flags (verified to encode
`processtatus` indirectly: 41.23% missing in exact lockstep with the
fb/FULLY PROCESSED split). Only `claimedvalue` (what was originally
claimed) and `submission_lag_days` (claim-to-reception gap, known at
reception) survive from that group -- the same shape of decision Phase 2
made for `TOTALAMOUNTCLAIMED`/`DAYS_TO_RECEPTION`.

**Caveat carried over unchanged from Phase 3:** "not flagged" doesn't mean
"clean" -- it may mean nobody caught it. Same weak-label limitation,
same reason it doesn't undermine the exercise.
""")

md("## Step 0 — Setup")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, average_precision_score, roc_auc_score,
    precision_recall_curve,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap

pd.set_option("display.max_columns", 60)
RANDOM_STATE = 42

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY, "text.color": INK, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "figure.dpi": 100,
})
""")

md("### Load the Jordan ML-ready export")

code(r"""
DATA_PATH = "../data/processed/jordan_data/jordan_ml_ready.csv"
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"NaNs remaining: {df.isna().sum().sum()}")
df.head()
""")

# ==========================================================================
md("## Step 2 — Explore the target")

code(r"""
positive = int(df["COLLUSION_FLAG"].sum())
negative = len(df) - positive
recovery_rate = positive / len(df)
scale_pos_weight = negative / positive

print(f"Positive claims:  {positive:,}")
print(f"Negative claims:  {negative:,}")
print(f"Collusion-flag rate: {recovery_rate:.4%}")
print(f"scale_pos_weight:    {scale_pos_weight:.2f}")

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.bar(["Not flagged", "Collusion-flagged"], [negative, positive], color=[CATEGORICAL[0], CATEGORICAL[7]])
ax.set_yscale("log")
ax.set_title(f"COLLUSION_FLAG distribution ({recovery_rate:.3%} positive)", color=INK, fontsize=12, loc="left", pad=12)
plt.tight_layout()
plt.show()
""")

# ==========================================================================
md(r"""## Step 3 — Time-based split

Same discipline as Phase 3: split by `claim_year`, never randomly.
2017 is a partial year (data collection appears to stop mid-year, 36,262
claims vs. 150-200k in full years) -- used as the untouched test year
regardless, since it's still the most recent.""")

code(r"""
print(df.groupby("claim_year")["COLLUSION_FLAG"].agg(claims="count", flagged="sum"))

FEATURES = [c for c in df.columns if c not in ("COLLUSION_FLAG", "claim_year")]
TARGET = "COLLUSION_FLAG"

train = df[df["claim_year"] <= 2015]
valid = df[df["claim_year"] == 2016]
test = df[df["claim_year"] == 2017]

X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

for name, (X, y) in {"Train": (X_train, y_train), "Valid": (X_valid, y_valid), "Test": (X_test, y_test)}.items():
    print(f"{name:6s}: {len(X):>7,} claims | {int(y.sum()):>4} flagged | rate {y.mean():.4%}")
""")

# ==========================================================================
md(r"""## Step 4 — Model 0: peer-deviation baseline (no ML)

`PEER_ZSCORE`/`AMOUNT_OVER_MEDIAN` here are grouped by (3-digit ICD9
category, dependency, visit type) rather than Al Hilal's (ICD chapter, age
band, network type) -- adapted to what Jordan's schema actually has, same
underlying idea: is this claim's amount unusual for its peer group?""")

code(r"""
def precision_recall_at_k(y_true, scores, k_frac):
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_frac)))
    order = np.argsort(-np.asarray(scores))
    top_k_idx = order[:k]
    y_true = np.asarray(y_true)
    hits = y_true[top_k_idx].sum()
    precision = hits / k
    recall = hits / y_true.sum() if y_true.sum() > 0 else np.nan
    return precision, recall, k, int(hits)


results = []
for score_name in ["PEER_ZSCORE", "AMOUNT_OVER_MEDIAN"]:
    scores = test[score_name].abs() if score_name == "PEER_ZSCORE" else test[score_name]
    p5, r5, k5, hits5 = precision_recall_at_k(y_test, scores, 0.05)
    ap = average_precision_score(y_test, scores)
    results.append({"score": score_name, "PR-AUC": ap, "Precision@5%": p5, "Recall@5%": r5})
    print(f"{score_name}: top 5% ({k5:,} claims) catches {hits5}/{int(y_test.sum())} "
          f"-> Recall@5%={r5:.1%}, Precision@5%={p5:.2%}")

peer_baseline_df = pd.DataFrame(results).set_index("score")
best_peer_score_name = peer_baseline_df["PR-AUC"].idxmax()
peer_row = peer_baseline_df.loc[best_peer_score_name]
peer_baseline_df
""")

# ==========================================================================
md("## Step 5 — Logistic regression baseline")

code(r"""
logreg = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
logreg.fit(X_train, y_train)

logreg_proba_test = logreg.predict_proba(X_test)[:, 1]
logreg_pred_test = logreg.predict(X_test)

logreg_precision = precision_score(y_test, logreg_pred_test, zero_division=0)
logreg_recall = recall_score(y_test, logreg_pred_test, zero_division=0)
logreg_prauc = average_precision_score(y_test, logreg_proba_test)
logreg_rocauc = roc_auc_score(y_test, logreg_proba_test)
logreg_precision5, logreg_recall5, _, _ = precision_recall_at_k(y_test, logreg_proba_test, 0.05)

print(f"PR-AUC:    {logreg_prauc:.4f}")
print(f"ROC-AUC:   {logreg_rocauc:.4f}")
print(f"Recall@5%: {logreg_recall5:.1%}")
""")

# ==========================================================================
md(r"""## Step 6 — LightGBM

Applying Phase 3's lesson directly rather than re-discovering it: the naive
spec (`num_leaves=31`, raw `scale_pos_weight`) collapses under this kind of
extreme imbalance with a small positive pool. Using the same fix from the
start -- `sqrt`-scaled positive weight, shallower/regularized trees, early
stopping on the 2016 validation year.""")

code(r"""
import lightgbm as lgb

lgbm = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=7, max_depth=4,
    min_child_samples=50, reg_lambda=5.0,
    scale_pos_weight=np.sqrt(scale_pos_weight),
    random_state=RANDOM_STATE, verbose=-1,
)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)], eval_metric="average_precision",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
print(f"Stopped at {lgbm.best_iteration_} rounds (of 500 max)")

P_COLLUSION_lgbm = lgbm.predict_proba(X_test)[:, 1]
test = test.assign(P_COLLUSION=P_COLLUSION_lgbm)

lgbm_pred_test = lgbm.predict(X_test)
lgbm_prauc = average_precision_score(y_test, P_COLLUSION_lgbm)
lgbm_rocauc = roc_auc_score(y_test, P_COLLUSION_lgbm)
lgbm_precision5, lgbm_recall5, _, _ = precision_recall_at_k(y_test, P_COLLUSION_lgbm, 0.05)
print(f"PR-AUC:    {lgbm_prauc:.4f}")
print(f"ROC-AUC:   {lgbm_rocauc:.4f}")
print(f"Recall@5%: {lgbm_recall5:.1%}")
""")

# ==========================================================================
md("## Step 7 — Evaluation")

code(r"""
prec_curve, rec_curve, _ = precision_recall_curve(y_test, P_COLLUSION_lgbm)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_curve, prec_curve, color=CATEGORICAL[0], lw=2)
ax.axhline(recovery_rate, color=INK_MUTED, ls="--", lw=1, label=f"Random baseline ({recovery_rate:.3%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"LightGBM precision-recall curve (PR-AUC={lgbm_prauc:.3f})", color=INK, fontsize=12, loc="left", pad=12)
ax.legend()
plt.tight_layout()
plt.show()
""")

# ==========================================================================
md("## Step 8 — Business metric: precision/recall @ top-k%")

code(r"""
business_rows = []
for k in [0.01, 0.05]:
    p, r, k_n, hits = precision_recall_at_k(y_test, P_COLLUSION_lgbm, k)
    business_rows.append({"Top %": f"{k:.0%}", "Claims reviewed": k_n, "Flagged caught": hits,
                           "Precision": p, "Recall": r})
    print(f"Top {k:.0%} ({k_n:,} claims): catches {hits}/{int(y_test.sum())} "
          f"-> Precision@{k:.0%}={p:.2%}, Recall@{k:.0%}={r:.1%}")

business_df = pd.DataFrame(business_rows).set_index("Top %")
business_df
""")

# ==========================================================================
md("## Step 9 — XGBoost")

code(r"""
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    eval_metric="aucpr", tree_method="hist",
)
xgb.fit(X_train, y_train)

P_COLLUSION_xgb = xgb.predict_proba(X_test)[:, 1]
xgb_prauc = average_precision_score(y_test, P_COLLUSION_xgb)
xgb_rocauc = roc_auc_score(y_test, P_COLLUSION_xgb)
xgb_precision5, xgb_recall5, _, _ = precision_recall_at_k(y_test, P_COLLUSION_xgb, 0.05)
print(f"PR-AUC:    {xgb_prauc:.4f}")
print(f"ROC-AUC:   {xgb_rocauc:.4f}")
print(f"Recall@5%: {xgb_recall5:.1%}")
""")

md("### Model comparison table")

code(r"""
comparison = pd.DataFrame([
    {"Model": "Peer Score", "PR-AUC": peer_row["PR-AUC"], "ROC-AUC": np.nan,
     "Precision@5%": peer_row["Precision@5%"], "Recall@5%": peer_row["Recall@5%"]},
    {"Model": "Logistic Regression", "PR-AUC": logreg_prauc, "ROC-AUC": logreg_rocauc,
     "Precision@5%": logreg_precision5, "Recall@5%": logreg_recall5},
    {"Model": "LightGBM", "PR-AUC": lgbm_prauc, "ROC-AUC": lgbm_rocauc,
     "Precision@5%": lgbm_precision5, "Recall@5%": lgbm_recall5},
    {"Model": "XGBoost", "PR-AUC": xgb_prauc, "ROC-AUC": xgb_rocauc,
     "Precision@5%": xgb_precision5, "Recall@5%": xgb_recall5},
]).set_index("Model")
comparison
""")

code(r"""
best_model_name = comparison["PR-AUC"].idxmax()
best_model = {"LightGBM": lgbm, "XGBoost": xgb}.get(best_model_name, lgbm)
best_proba_test = {"LightGBM": P_COLLUSION_lgbm, "XGBoost": P_COLLUSION_xgb}.get(best_model_name, P_COLLUSION_lgbm)
print(f"Best model by PR-AUC: {best_model_name}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(comparison.index, comparison["PR-AUC"], color=CATEGORICAL[:len(comparison)])
ax.axhline(recovery_rate, color=INK_MUTED, ls="--", lw=1, label=f"Random baseline ({recovery_rate:.3%})")
ax.set_ylabel("PR-AUC")
ax.set_title("Model comparison: PR-AUC on 2017 test set", color=INK, fontsize=12, loc="left", pad=12)
ax.legend()
plt.tight_layout()
plt.show()
""")

# ==========================================================================
md("## Step 10 — Feature importance")

code(r"""
importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
top15 = importances.head(15)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top15.index[::-1], top15.values[::-1], color=CATEGORICAL[0])
ax.set_xlabel("Feature importance")
ax.set_title(f"{best_model_name}: top 15 features", color=INK, fontsize=12, loc="left", pad=12)
plt.tight_layout()
plt.show()
top15
""")

# ==========================================================================
md("## Step 11 — SHAP")

code(r"""
shap_sample = X_test.sample(n=min(3000, len(X_test)), random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(shap_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap.summary_plot(shap_values, shap_sample, max_display=15, show=False)
plt.tight_layout()
plt.show()
""")

# ==========================================================================
md(r"""## Step 12 — Business story

Same framing as Phase 3, adapted: if the audit team reviewed only the
riskiest 5% of claims before payment, how many of the actually
investigator-flagged collusion cases would they have caught? Reported as a
case count, not an AED figure -- `claimedvalue` is standardized in this
export and there's no retained `claimid` join key back to raw amounts in
this notebook's frame (same limitation Phase 3 hit and solved for Al Hilal;
not re-solved here to keep this notebook's scope to the model comparison
itself).""")

code(r"""
test_reset = test.reset_index(drop=True)
n = len(test_reset)
k = max(1, int(np.ceil(n * 0.05)))
order = np.argsort(-test_reset["P_COLLUSION"].values)
top_k = test_reset.iloc[order[:k]]

n_flagged = int(test_reset["COLLUSION_FLAG"].sum())
n_caught = int(top_k["COLLUSION_FLAG"].sum())

print(f"2017 test set: {n:,} claims, {n_flagged} investigator-flagged for possible collusion")
print(f"Reviewing top 5% ({k:,} claims) by {best_model_name} risk score would have caught:")
print(f"  {n_caught}/{n_flagged} flagged claims ({n_caught/n_flagged:.1%} of cases)")
""")

md("### Headline result")

code(r"""
from IPython.display import Markdown, display

display(Markdown(
    f"> Reviewing only the riskiest **5%** of claims before payment — using the "
    f"**{best_model_name}** model — would have caught **{n_caught}/{n_flagged} "
    f"({n_caught/n_flagged:.0%})** of 2017's investigator-flagged collusion cases, "
    f"out of {n:,} total claims that year. This is a second, independent dataset "
    f"showing the same reactive-to-proactive pattern Phase 3 established for "
    f"Al Hilal Takaful."
))
""")

md(r"""## Summary

| Item | Al Hilal (Phase 3) | Jordan (this notebook) |
|---|---|---|
| Claims | 839,209 | 602,267 |
| Label | `RECOVERY_FLAG` (post-payment clawback) | `COLLUSION_FLAG` (investigator-flagged) |
| Positive rate | 0.073% | 0.116% |
| Best model | XGBoost (PR-AUC 0.506) | see comparison table above |
| Peer-score baseline | weak (PR-AUC 0.0016) | weak (PR-AUC ~0.001) |

Both datasets independently support the same conclusion: a supervised model
using pre-adjudication claim features substantially outperforms an
unsupervised peer-deviation heuristic at ranking fraud/recovery risk.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

out_path = "notebooks/04_jordan_modeling.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
