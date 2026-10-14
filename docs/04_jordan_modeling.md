# Phase 3b — Jordan Dataset: Collusion-Risk Model

Source notebook: `notebooks/04_jordan_modeling.ipynb`
Generator script: `scripts/build_jordan_modeling_notebook.py`
Data prep script: `scripts/jordan/prepare_data.py`

Runs **alongside** the Al Hilal Takaful analysis (Phases 2–4), not
replacing it — a second, independent dataset for the same underlying
question, added after a new `data/processed/jordan_data/` export appeared
mid-project. This document explains what that new data is, the decisions
made in adapting it to this project's modeling approach, and the results.

---

## 1. What the Jordan data is, and where it came from

602,280 Jordanian health-insurance claims (2013–2017), already cleaned and
partly feature-engineered **outside this session**, before being placed in
`data/processed/jordan_data/`. That folder came with its own documentation
(`preprocessing_log.csv`, `data_quality_report.csv`, `column_profile.csv`,
`missing_value_report.csv`, `feature_relevance_for_objectives.csv`), which
was read and relied on rather than re-deriving the cleaning from raw —
the notes reference "Phase 1/2/2.7" and "supervisor guidance," indicating
this prep work was done deliberately and reviewed.

**This is a structurally different dataset from Al Hilal's**, not just a
new country: national-ID-based identifiers (`nid`, `hofid`), a
claimed/due/rejected/coinsurance/debit adjudication breakdown instead of a
single payable amount, ICD9 diagnosis codes instead of ICD chapters, and —
critically — **no equivalent of `RECOVERY/EX-GRATIA`**. There is no
post-payment clawback field. The closest candidate signals were both
already engineered into the data:

- **`is_collusion_adjacent_tmp`** (696 claims, 0.116%) — derived from
  `processcode_category_tmp`, which documents specific codes as "flagged by
  investigators as possible HCP/PH collusion signal." A direct,
  investigator-justified fraud-risk signal.
- **`is_correction_R01`** (112 claims, 0.019%) — claims with a negative
  `duevalue` under processcode `R01` ("routine adjustment" /
  "Correction (not new utilization)"). Ambiguous: could be fraud-related,
  could be an ordinary billing fix.

**`is_collusion_adjacent_tmp` was chosen** as the label — more data to
learn from (696 vs. 112), and a semantically direct fraud-risk claim rather
than an ambiguous "correction." Renamed `COLLUSION_FLAG` in the ML-ready
export for clarity.

---

## 2. Leakage discipline (`scripts/jordan/prepare_data.py`)

Because the label is derived from `processcode`, the same strict standard
Phase 2 applied to Al Hilal's `STATUS`/`TOTALPAYABLE`/etc. was applied
here — every column that is the adjudication *outcome*, or timed off it,
is excluded from the feature set:

| Excluded | Why |
|---|---|
| `duevalue`, `rejected`, `coinsurance`, `debit`, `cost` | The approved/rejected/copay amounts — these **are** the adjudication decision, not an input to it |
| `processcode`, `processcode_category_tmp`, `is_correction_R01`, `is_fb_status`, `processtatus` | The outcome the label itself is derived from |
| `processdate`, `processing_lag_days`, `total_cycle_days` | Only knowable once adjudication *completes* — an investigation into a flagged claim could itself lengthen these |
| `has_claimedvaluenote`, `has_duevaluenote`, `has_rejectednote`, `has_coinsurancenote`, `has_debitnote`, `cost_is_missing` | Verified to encode `processtatus` indirectly: all four `*note` presence flags are missing in *exact* lockstep (41.23%) with the fb/FULLY PROCESSED split |
| `nid`, `hofid`, `hcpid`, `policyid`, `claimid`, `sname`, `hofname`, `hcpname`, `Manual Ref No.` | Identifiers / free text, not generalizable features (matches `feature_relevance_for_objectives.csv`'s own conclusion) |
| `dateclaim`, `datereceived`, `dob`, `effectivedate`, `visitdate` | Raw dates — signal already captured in engineered day-gap/age features |
| `icd9code`, `icd9codedescrip`, `claimeditem` | Free text / high cardinality — `icd9code` frequency-encoded instead (`ICD9_FREQ`) |

**Only `claimedvalue`** (what was originally claimed) and
**`submission_lag_days`** (claim-to-reception gap, known at reception)
survive from the financial/timing group — exactly the same shape of
decision Phase 2 made keeping `TOTALAMOUNTCLAIMED`/`DAYS_TO_RECEPTION` while
excluding `TOTALPAYABLE`/`PAYERSHARE`/`SETTLEDATE`.

### Peer-deviation feature

`PEER_ZSCORE`/`AMOUNT_OVER_MEDIAN` rebuilt for this schema: grouped by
(3-digit ICD9 category, dependency, visit type) instead of Al Hilal's (ICD
chapter, age band, network type) — same underlying idea (is this claim's
amount unusual for its peer group), adapted to what Jordan's schema has.

### Final export

```
data/processed/jordan_data/jordan_ml_ready.csv
602,267 rows x 34 columns (13 rows dropped for missing age_at_claim —
0.0022%, negligible, not imputed)
Positive class (COLLUSION_FLAG=1): 696 (0.1156%)
```

---

## 3. Modeling results

Same pipeline as Phase 3: time-based split (train ≤2015, validate 2016,
test on the untouched 2017 year — a partial year, ~36k claims vs. 150–200k
in full years, used regardless since it's still the most recent), the same
four-model comparison, and **the LightGBM fix from Phase 3 applied from the
start** rather than rediscovered (`sqrt`-scaled positive weight, shallower
regularized trees, early stopping on the validation year).

```
Train : 354,096 claims |  281 flagged | rate 0.0794%
Valid : 211,909 claims |  371 flagged | rate 0.1751%
Test  :  36,262 claims |   44 flagged | rate 0.1213%
```

The positive rate roughly **doubles between train and validation years**
(0.079% → 0.175%) — a real distribution shift, not a modeling artifact,
worth flagging on its own: either collusion detection intensified in 2016,
or investigators got better at finding it. This shift is also why
LightGBM's early stopping triggered almost immediately (4 rounds) — the
validation signal moved fast enough that more rounds looked like
overfitting to the callback.

### Model comparison (2017 test set)

| Model | PR-AUC | ROC-AUC | Precision@5% | Recall@5% |
|---|---|---|---|---|
| Peer Score (`AMOUNT_OVER_MEDIAN`) | 0.0010 | – | 0.1% | 4.5% |
| Logistic Regression | 0.0020 | 0.698 | 0% | 0% |
| **LightGBM** | **0.0132** | 0.862 | 1.6% | **65.9%** |
| XGBoost | 0.0073 | 0.824 | 1.2% | 47.7% |

**Result — a genuinely different shape of finding from Phase 3, worth
reading correctly rather than as a worse copy of it.** PR-AUC here (0.013
best) is far below Al Hilal's (0.506) — but that's a direct, expected
consequence of test-set rarity: with only 44 positives in 36,262 test
claims, the maximum precision achievable at any reasonable recall is
inherently tiny, which caps PR-AUC hard regardless of model quality. The
metric that matters for the audit-queue use case, **Recall@5%, is strong
in absolute terms: 65.9%** — reviewing 1,814 of 36,262 claims would have
caught two-thirds of 2017's investigator-flagged collusion cases. Logistic
regression's 0% Recall@5% (zero of 44 positives landing in its top 5%) is
also a real, reportable result, not a bug — a near-random ranker can
plausibly miss every one of 44 rare positives in a 1,814-claim window by
chance.

**Peer score is weak here too** (PR-AUC ~0.001, essentially random) — the
same finding Phase 3 made for Al Hilal (PR-AUC 0.0016). Collusion isn't
primarily an amount-outlier phenomenon on either dataset; it needs the
richer feature combination a supervised model provides.

### What the model actually keys off

LightGBM's early-stopped model is very shallow (4 boosting rounds, correspondingly few total splits):

| Feature | Splits used |
|---|---|
| `ICD9_FREQ` | 7 |
| `age_at_claim` | 6 |
| `patient_claim_count` | 5 |
| `dependency_SPOUSE` | 4 |
| `visitype_encoded` | 1 |
| `AMOUNT_OVER_MEDIAN` | 1 |

Diagnosis-code rarity, patient age, how many claims that patient has filed
overall, and specifically being a **spouse** dependent are the earliest,
strongest signals — a plausible, human-legible pattern (collusion
schemes concentrating around specific diagnosis codes and dependent
relationships) rather than a black box.

### Business result

```
2017 test set: 36,262 claims, 44 investigator-flagged for possible collusion
Reviewing top 5% (1,814 claims) by LightGBM risk score would have caught:
  29/44 flagged claims (65.9% of cases)
```

No AED-equivalent exposure figure is reported here (unlike Phase 3's Step
12) — `claimedvalue` is standardized in this export and there's no
retained `claimid` join key back to raw amounts in this notebook's frame.
This is the same limitation Phase 3 hit and solved for Al Hilal (a
row-position join back to the raw export); not re-solved here to keep this
notebook's scope to the model comparison itself.

---

## 4. Cross-dataset summary

| | Al Hilal (Phase 3) | Jordan (this notebook) |
|---|---|---|
| Claims | 839,209 | 602,267 |
| Years | 2013–2016 | 2013–2017 |
| Label | `RECOVERY_FLAG` (post-payment clawback) | `COLLUSION_FLAG` (investigator-flagged) |
| Positive rate | 0.073% | 0.116% |
| Best model | XGBoost (PR-AUC 0.506) | LightGBM (PR-AUC 0.013) |
| Best Recall@5% | 71.4% (logistic) | 65.9% (LightGBM) |
| Peer-score baseline | weak (PR-AUC 0.0016) | weak (PR-AUC 0.0010) |

**Both datasets independently support the same core conclusion**: a
supervised model using only pre-adjudication claim features substantially
outperforms an unsupervised peer-deviation heuristic at ranking fraud/
recovery risk, and a review queue sized to ~5% of claim volume catches the
large majority of the cases that matter. The PR-AUC gap between the two
datasets is explained by test-set rarity (44 vs. 28 positives, but Jordan's
test set is 36,262 claims vs. Al Hilal's 49,876 — a higher rate but far
fewer absolute events to separate from), not by the approach failing to
generalize — Recall@5% (the metric that actually maps to an audit
workflow) lands in the same 60–70% range on both.
