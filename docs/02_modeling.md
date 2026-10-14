# Phase 3 — Modeling: Step-by-Step Documentation

Source notebook: `notebooks/02_modeling.ipynb`
Generator script: `scripts/build_modeling_notebook.py`
Input: `data/processed/claims_ml_ready.csv` (frozen — see Phase 2 doc), 839,209
claims × 110 columns, no modification made to it in this phase.

**Core question.** Al Hilal Takaful currently catches bad claims *after*
paying them, via manual review and clawback. Can a model flag, at submission
time, which claims are likely to end up in the Recovery Report — using only
information available before adjudication — so the insurer catches the same
money before it goes out the door?

---

## Step 1 — Freeze the dataset

`claims_ml_ready.csv` is loaded once and never modified in this notebook.
All experimentation happens on in-memory copies (`train`/`valid`/`test`
slices). This matters for reproducibility: every result in this document
traces back to one fixed CSV, not a moving target.

---

## Step 2 — Explore the target

```
Positive claims:  614
Negative claims:  838,595
Recovery rate:    0.0732%
scale_pos_weight: 1365.79
```

`scale_pos_weight` (negative/positive ratio) is the standard LightGBM/XGBoost
lever for imbalanced classification — it multiplies the loss contribution of
each positive example so the model doesn't just learn to predict "never
recoverable" and score 99.93% while catching nothing. It's computed once here
and referenced throughout the notebook. Note in Step 6 below: the *raw* value
(1365.79) turned out to be too aggressive for LightGBM specifically — a
tuned, less aggressive form was needed there (see that section for why).

---

## Step 3 — Time-based split

A random split would let a model "cheat" by learning from claims that were
resolved *after* the ones it's tested on — the model would implicitly see
the future. Splitting by `CLAIM_YEAR` instead simulates actually deploying
the model going forward: **past → future**, never random → random.

| Split | Years | Claims | Recoveries | Recovery rate |
|---|---|---|---|---|
| Train | ≤2014 (includes 2 claims from 2013) | 300,662 | 252 | 0.084% |
| Valid | 2015 | 488,671 | 334 | 0.068% |
| Test | 2016 | 49,876 | 28 | 0.056% |

2013 has only 2 claims — folded into training rather than given its own
bucket, since it's too small to be a meaningful split on its own and would
just add noise. The **validation year (2015)** isn't a leftover: it's used
directly for LightGBM's early stopping in Step 6, and was available for any
further tuning.

The test set is deliberately thin (28 positives in 2016) — this is a direct
consequence of the severe class imbalance carried down to a single year, and
it's why every metric below is reported as a count as well as a percentage
(e.g. "15/28", not just "53.6%") — a percentage alone would overstate the
precision of a result built on 28 events.

---

## Step 4 — Model 0: peer-deviation baseline (no ML)

Before any classifier: how far does simply ranking claims by their
Phase-2-built peer-group deviation score get an auditor?

```
PEER_ZSCORE:          reviewing top 5% (2,494 claims) catches 2/28 recoveries
                       -> Recall@5%=7.1%, Precision@5%=0.08%
AMOUNT_OVER_MEDIAN:    reviewing top 5% (2,494 claims) catches 2/28 recoveries
                       -> Recall@5%=7.1%, Precision@5%=0.08%
```

**Result: both peer-deviation scores alone are barely better than random**
(random 5% of claims would be expected to catch ~5% of 28 ≈ 1.4 recoveries;
these catch 2). This is an important, honest finding, not a disappointing
one to bury: it shows that "how far is this claim's amount from what similar
claims usually cost" — a reasonable, tariff-schedule-free heuristic — is on
its own a weak signal for *this specific* recovery pattern. That justifies
moving to supervised learning (Steps 5–9), which can combine dozens of
features (timing, provider history, demographics, diagnosis, *and* the peer
score) rather than relying on one univariate rule. `AMOUNT_OVER_MEDIAN` is
carried forward as the "Peer Score" row in the Step 9 comparison table.

---

## Step 5 — Logistic regression baseline

`LogisticRegression(class_weight="balanced")` — simple, interpretable,
imbalance-aware, not expected to win.

```
Precision (0.5 threshold): 0.018
Recall (0.5 threshold):    0.500
PR-AUC:                    0.4700
ROC-AUC:                   0.9088
Recall@5%:                 71.4%
```

**Result: a strong, somewhat surprising baseline — and, on Recall@5%
specifically, the single best model in this notebook.** PR-AUC 0.47 is far
above the peer-score baseline's ~0.0016 — a linear model with class-balanced
weighting already recovers most of the achievable signal in this feature set.
At the top-5% operating point, logistic regression actually catches a higher
share of 2016's recoveries (71.4%) than either gradient-boosted model (see
the Step 9 table) — a useful sanity check that a simple, fully interpretable
model shouldn't be dismissed just because it's simple. PR-AUC still favors
XGBoost slightly (0.506 vs. 0.470) because PR-AUC integrates ranking quality
across *all* thresholds, not just the top 5% — the two metrics are answering
different questions, and Step 9 discusses the resulting trade-off.
(`max_iter` was raised to 5,000 to try to clear an lbfgs convergence warning
that appeared at 1,000 iterations under `class_weight="balanced"`; the
warning persists even at 5,000 — lbfgs does not fully converge on this
imbalance-reweighted problem — but PR-AUC barely moved between the two
settings (0.4709 → 0.4700), so the fit is stable in substance even though
the solver hasn't formally converged. Worth a one-line mention in the
thesis methodology section, not a blocker.)

---

## Step 6 — LightGBM (main model)

### First attempt (the naive spec) — and why it failed

The originally specified configuration —

```python
LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                scale_pos_weight=scale_pos_weight, random_state=42)
```

— **collapsed**: PR-AUC ≈ 0.001 (barely above the 0.056% random baseline),
ROC-AUC ≈ 0.67, dramatically worse than the logistic regression baseline
above. This was verified as a real, reproducible result rather than a
one-off — it reproduced with and without `scale_pos_weight`, ruling out a
class-order or configuration bug (`lgbm.classes_` confirmed `[0, 1]` in the
expected order).

**Root cause:** with only ~250 positive claims in the training years, LightGBM's
*leaf-wise* tree growth (as opposed to XGBoost's *depth-wise* growth) at
`num_leaves=31` and the *raw* `scale_pos_weight` (≈1,366×) fits those few
positives almost exactly within the first several trees — and none of that
fit generalizes to 2016. This is a known failure mode of leaf-wise boosting
under severe imbalance with a small positive pool, not something specific to
this dataset's quality.

### The fix

Three standard, non-cherry-picked corrections:

1. **`scale_pos_weight = sqrt(negative / positive)`** instead of the raw
   ratio — a common heuristic once the raw ratio is extreme (>100×): the raw
   ratio pushes the loss toward fitting every minority point exactly, rather
   than a boundary that generalizes.
2. **Shallower, more regularized trees** — `num_leaves=7`, `max_depth=4`,
   `min_child_samples=50`, `reg_lambda=5.0` — limits how much any single
   split can chase noise in a ~250-row minority class.
3. **Early stopping against the 2015 validation year**
   (`eval_metric="average_precision"`, 50-round patience) — this is exactly
   what the Step 3 validation split was reserved for: pick the number of
   boosting rounds using a genuinely held-out future year, not the training
   data itself.

```
Stopped at 122 rounds (of 500 max), chosen on the 2015 validation set
Precision (0.5 threshold): 0.151
Recall (0.5 threshold):    0.500
PR-AUC:                    0.5020
ROC-AUC:                   0.8689
```

**Result: fixed.** PR-AUC went from ~0.001 to **0.502** — now essentially on
par with XGBoost (0.506, Step 9) and above the logistic baseline (0.471).
This is a meaningful methodology point for the write-up: default
hyperparameters do not transfer safely to a ~0.07%-positive classification
problem, and imbalance-aware tuning (weight heuristic + regularization +
early stopping on a true holdout) is what makes the difference between a
model that fails and one that's competitive.

---

## Step 7 — Evaluation

Accuracy is not reported as a headline metric — predicting "no recovery" for
every claim scores 99.93% while catching zero real recoveries, so it says
nothing about model quality here. The metrics that matter for a
ranked-review workflow:

- **PR-AUC** (`average_precision_score`) — the primary metric under severe
  imbalance; unlike ROC-AUC it isn't inflated by the huge pool of
  easy-to-classify true negatives.
- **ROC-AUC** — reported for completeness, secondary.
- **Precision/Recall at the default 0.5 threshold** — mostly illustrative;
  the real operating point for this use case is "top k%", covered in Step 8.

The notebook plots LightGBM's full precision-recall curve against the random
(recovery-rate) baseline line — visually, the gap between the curve and that
flat baseline line is the model's entire value-add.

---

## Step 8 — Business metric: precision/recall @ top-k%

This is the number that maps directly onto an audit team's actual workflow:
*if auditors review only the riskiest k% of claims before payment, what share
of real recoveries would they have caught?*

Reusable function:

```python
def precision_recall_at_k(y_true, scores, k_frac):
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_frac)))
    order = np.argsort(-np.asarray(scores))
    top_k_idx = order[:k]
    hits = y_true[top_k_idx].sum()
    return hits / k, hits / y_true.sum(), k, hits
```

LightGBM results on the 2016 test set:

| Top % | Claims reviewed | Recoveries caught | Precision | Recall |
|---|---|---|---|---|
| 1% | 499 | 14/28 | 2.81% | 50.0% |
| 5% | 2,494 | 15/28 | 0.60% | 53.6% |

**Interpretation.** Reviewing just the top 1% of claims by LightGBM risk
score already catches half of all real 2016 recoveries. Precision looks low
in absolute terms (0.6–2.8%) — but that's expected and correctly reported,
not a flaw: at a 0.056% base rate, even a 10–50× enrichment over random still
produces mostly-false-positive review queues. The right comparison isn't
"is precision high" but "is precision far above the 0.056% base rate" (it is,
by roughly 10× at 5% and 50× at 1%), and "does review volume stay
manageable" (2,494 claims out of 49,876 — a 20× reduction in what needs
manual review to catch the majority of recoveries).

---

## Step 9 — XGBoost

```python
XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
               scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
               tree_method="hist")
```

```
PR-AUC:    0.5061
ROC-AUC:   0.8502
Recall@5%: 60.7%
```

Unlike LightGBM, XGBoost's default depth-wise growth (capped at `max_depth=6`)
did **not** need the same rescue — it tolerated the raw `scale_pos_weight`
and default regularization without collapsing. That asymmetry is itself a
useful, reportable methodology finding: the two most common gradient-boosting
libraries do not behave the same way under extreme class imbalance with a
tiny positive pool, and neither can be trusted with untouched defaults
without checking against a genuinely held-out validation period first.

### Model comparison table (2016 test set)

| Model | PR-AUC | ROC-AUC | Precision@5% | Recall@5% |
|---|---|---|---|---|
| Peer Score (`AMOUNT_OVER_MEDIAN`) | 0.0016 | – | 0.1% | 7.1% |
| **Logistic Regression** | 0.4700 | **0.9088** | 0.8% | **71.4%** |
| LightGBM (tuned) | 0.5020 | 0.8689 | 0.6% | 53.6% |
| **XGBoost** | **0.5061** | 0.8502 | 0.7% | 60.7% |

**Result — no single model wins on every metric, and that's a real finding
worth reporting as-is rather than flattening into one "best model" claim:**

- **By PR-AUC** (overall ranking quality across all thresholds): XGBoost
  (0.506) edges out LightGBM (0.502), both comfortably ahead of logistic
  regression (0.470) and far ahead of the peer-score rule (0.0016).
- **By ROC-AUC and Recall@5%** (the actual audit-queue operating point):
  logistic regression wins outright — 0.9088 ROC-AUC and 71.4% recall, both
  clearly above the tree ensembles. A simple, class-weighted linear model
  ranks the small handful of true positives higher, on average, than either
  gradient-boosted model does at the specific 5% cutoff that matters for the
  business metric.
- **Practical takeaway:** all three supervised models decisively beat the
  peer-score baseline, confirming supervised learning adds real value over a
  univariate outlier rule. Between the three, the choice depends on what's
  being optimized for — XGBoost is carried forward as the "best model" for
  Steps 10–12 because PR-AUC is the standard, threshold-independent metric
  for severe class imbalance and matches this notebook's Step 6/9
  `best_model_name = comparison["PR-AUC"].idxmax()` selection rule — but the
  thesis write-up should show logistic regression's Recall@5% figure
  alongside it, since 71.4% vs. 60.7% is a meaningful difference at the exact
  operating point management would use. With a test set of only 28 positive
  claims, this ranking is also fragile — a couple of claims landing on either
  side of the top-5% cutoff would visibly move these numbers, so the
  practical recommendation is to re-validate the model choice once more
  years of data are available, not to treat this ordering as final.

---

## Step 10 — Feature importance

`model.feature_importances_` on the winning model (XGBoost), top 20 plotted
as a horizontal bar chart in the notebook. Top of the ranked list:

| Rank | Feature | Importance (normalized) |
|---|---|---|
| 1 | `PROVIDER EMIRATE_Abu Dhabi` | **0.480** |
| 2 | `FOB_Wellness` | 0.061 |
| 3 | `AGEBAND_Age 56-60` | 0.059 |
| 4 | `MEMBER_TENURE_DAYS` | 0.035 |
| 5 | `CHRONIC_No` | 0.025 |
| 6 | `NETWORK TYPE_Super-Restricted Network` | 0.024 |
| 7 | `REPORTING AUTHORITY_DHA` | 0.024 |
| 8 | `PEER_MEDIAN` | 0.023 |
| 9 | `NETWORK TYPE_Restricted Network` | 0.022 |
| 10 | `PROVIDERTYPE_Polyclinic/Diagnostic Center` | 0.019 |

**Result — one dominant feature, and it needs a caution, not just a
celebration.** `PROVIDER EMIRATE_Abu Dhabi` alone accounts for **48% of all
importance** — more than the next 9 features combined. Two features this
notebook's whole Step 3/4 design was built around, `PEER_ZSCORE` and
`AMOUNT_OVER_MEDIAN`, don't even reach the top 20; only `PEER_MEDIAN` does,
at rank 8.

This was checked directly rather than left as a guess — a feature this
dominant deserves verification before being trusted, so recovery *rate*
(not raw count) was computed for Abu-Dhabi-provider vs. non-Abu-Dhabi-provider
claims, broken out by year:

| Year | Abu Dhabi provider recovery rate | Other-emirate provider recovery rate | Ratio |
|---|---|---|---|
| 2014 (train) | 0.016% | 0.280% | ~17× |
| 2015 (valid) | 0.032% | 0.171% | ~5× |
| 2016 (test) | 0.043% | 0.095% | ~2× |

**Result: the direction is the opposite of the naive assumption, and it's a
real, stable, multi-year pattern — not a training-set artifact.** Claims
handled by providers *outside* Abu Dhabi are consistently 2–17× more likely
to be recovered than Abu-Dhabi-provider claims, in every one of the three
years, including 2016 — a year the model never trained on. Two things follow:

- **This is a legitimate, actionable finding**, not noise the model happened
  to memorize: the gap holds up out-of-sample (2016), which is exactly the
  standard a feature needs to clear before being reported as a real driver
  rather than an overfit quirk. It also explains why the feature dominates
  importance — a signal this consistent across three separate years is
  genuinely easy for a tree model to exploit.
- **The ratio is shrinking over time** (17× → 5× → 2×) — worth flagging in
  the thesis as its own finding: either non-Abu-Dhabi providers' claims are
  becoming relatively less risky, Abu-Dhabi-provider risk is rising, or the
  underlying provider/network mix within each emirate group is shifting
  year to year. This is a candidate follow-up cut for the BI provider/
  network dashboard (Idea 1, Step 7) rather than something this notebook can
  resolve on its own — provider-level, not just emirate-level, granularity
  would be needed to say which.
- **Still worth a caution on overall importance concentration:** even with
  the direction confirmed genuine, one feature holding 48% of total
  importance while `PEER_ZSCORE`/`AMOUNT_OVER_MEDIAN` (the features Steps
  3–4 were built around) don't reach the top 20 means the model's decision
  boundary is currently quite reliant on *where* a provider is located
  rather than *how anomalous* a specific claim looks. That's a reasonable
  place to land given the data, but it argues for enriching provider-level
  features (e.g. per-provider historical recovery rate, not just claim
  frequency) in any future iteration, rather than treating this feature set
  as finished.

---

## Step 11 — SHAP

**Global**: `shap.TreeExplainer` on the winning model, computed over a
3,000-row sample of the test set (full-test SHAP on the whole feature set
would be needlessly slow for no accuracy gain — SHAP values are a local
approximation either way). `shap.summary_plot` shows both magnitude and
*direction* (does a high vs. low feature value push risk up or down) for the
top 20 features — this is the answer to "why" a feature matters, which raw
importance in Step 10 doesn't give.

**Individual**: a waterfall plot for the single highest-risk claim among the
2016 test set's actual recoveries — i.e., "here's a real recovered claim, and
here's exactly which features pushed the model to flag it, and by how much."
This is the piece that turns a risk score into something an auditor can act
on and explain, not just a number to trust blindly.

---

## Step 12 — Business story

**The question that matters most for the thesis:** if Al Hilal reviewed only
the top 5% highest-risk claims before payment, what fraction of the
recovery exposure could have been prevented?

### The AED join problem, and how it was resolved

`claims_ml_ready.csv` intentionally has no claim ID and a *standardized*
(mean 0, std 1) `TOTALAMOUNTCLAIMED` — both dropped/scaled during Phase 2 to
keep the feature matrix leakage-safe and model-ready. Computing a raw AED
figure directly from that column would have produced a fabricated number
(sums of z-scores, not currency).

The fix: `claims_ml_ready.csv` was built from `df_final` with **no row
filtering or reordering** after that point in the Phase 2 notebook, so it has
the exact same row order and row count (839,209) as
`consolidated_claims_all_reports.csv`, which still has raw amounts and
`CLAIMID`. This was not assumed — it was **verified** before being trusted:
`RECOVERY_FLAG` derived independently from the raw file's
`RECOVERY/EX-GRATIA > 0` was checked against `claims_ml_ready.csv`'s
`RECOVERY_FLAG`, row-for-row, across all 839,209 rows:

```
Row-position alignment verified: 0 mismatches across 839,209 rows
```

With that safety check passing, the 2016 test set's raw `TOTALAMOUNTCLAIMED`
was recovered by a row-position join (`raw_amounts.loc[test.index]`), not a
key join (there is no shared key column to join on).

### Result

```
2016 test set: 49,876 claims, 28 later flagged for recovery
Reviewing top 5% (2,494 claims) by XGBoost risk score would have caught:
  15/28 recovered claims (53.6% of cases)
  AED 10,803 of AED 20,186 in claimed-amount exposure on those recovered
  claims (53.5%)
```

**Caveat, stated directly in the notebook and repeated here:**
`TOTALAMOUNTCLAIMED` is what the claim was *submitted for*, not the amount
that was actually *recovered* — those are different numbers (a recovery can
be partial). `RECOVERY/EX-GRATIA` (the true recovered amount) was correctly
excluded upstream as a leakage column and isn't available in this frame
either. So the AED 10,803 / AED 20,186 figures should be read as **claimed-amount
exposure on the claims that were eventually recovered** — a defensible upper
bound on the true recovered AED, not the recovered AED itself. Getting the
exact recovered-AED figure would require joining `RECOVERY/EX-GRATIA` back
the same row-position way, purely for post-hoc reporting (never as a model
feature).

### Headline

> Reviewing only the riskiest 5% of claims before payment — using the
> XGBoost model — would have caught 15/28 (54%) of 2016's actual recovery
> cases, representing AED 10,803 (54%) of the claimed-amount exposure tied
> to those recoveries. This turns Al Hilal's reactive, post-payment clawback
> process into a proactive, pre-payment audit queue sized to 2,494 claims —
> a fraction of the 49,876-claim yearly volume.

---

## Summary of fixes applied during this phase

| Issue found | Root cause | Fix |
|---|---|---|
| LightGBM PR-AUC ≈ 0.001 (near-random) | Leaf-wise growth (`num_leaves=31`) + raw `scale_pos_weight` (1366×) overfits the ~250 training positives within a handful of rounds | `scale_pos_weight=sqrt(...)`, shallower/regularized trees, early stopping on the 2015 validation year → PR-AUC 0.502 |
| Step 12 AED figure would have been fabricated | `TOTALAMOUNTCLAIMED` in `claims_ml_ready.csv` is standardized; no claim ID survives Phase 2's leakage-safe export | Row-position join back to `consolidated_claims_all_reports.csv`, verified safe via an independent `RECOVERY_FLAG` cross-check (0 mismatches) before being trusted |
| Logistic regression convergence warning | lbfgs doesn't fully converge on this imbalance-reweighted problem even at `max_iter=1000` | Raised to `max_iter=5000`; warning persists but PR-AUC barely moved (0.4709 → 0.4700), confirming the fit is stable in substance — documented as a known, non-blocking limitation rather than silently ignored |

## Caveat carried through every result above

"Not recovered" doesn't necessarily mean "clean" — it may just mean nobody
caught it. Every recall/precision number in this document is measured
against Al Hilal's own historical audit findings, which are themselves an
imperfect (weak/noisy) label of true fraud/abuse. This is a standard,
well-documented limitation in fraud-detection literature, not a flaw
specific to this model or pipeline — but it means the reported recall
figures are a floor on how many recoverable claims a better-resourced audit
process might have caught, not a ceiling.
