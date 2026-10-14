# Phase 2 — Data Processing: Step-by-Step Documentation

Source notebook: `notebooks/data_processing.ipynb`
Generator script: `scripts/build_notebook.py`

This document explains, in order, everything the processing pipeline does to turn
the raw Al Hilal Takaful Excel exports into `data/processed/claims_ml_ready.csv`
— the frozen dataset used for modeling in Phase 3 — and why each step exists.

---

## 1. Inputs

| Folder | Files | Content |
|---|---|---|
| `data/raw/utilization/` | 4 workbooks (`UR UY-2014 - NAS.xlsx`, `UTILIZ as at JAN 2014.xlsx`, `Utilization Report - UY-2014 Al Hilal Takaful.xlsx`, `Utilization Report - UY-2015 Al Hilal Takaful.xlsx`) | Claims data — one `Data` sheet, one `Recovery` sheet, one `PendingAuthorizations` sheet per workbook, plus assorted reference sheets (Cover Page, Definitions, etc.) |
| `data/raw/beneficiaries/` | 5 workbooks (`AHT 2014 A/B.xlsx`, `AHT 2015 A/B.xlsx`, `AHT 2016 A.xlsx`) | Beneficiary/policy data — one `Sheet1`/`Data` sheet per workbook |

The raw exports are not clean tables: each sheet has a handful of title rows
above the real header row, some workbooks carry a corrupted Excel "defined
name" that crashes `openpyxl` on read, and several reference sheets (Cover
Page, Provider and Group, Definitions, …) have no tabular header at all.

---

## 2. Loading (`notebook §§1–4`)

**`strip_defined_names`** — some workbooks have a broken named-range
definition that crashes `openpyxl`. The fix is a temp copy of the file with
the defined-names XML entry stripped, not a re-export of the whole workbook.

**`detect_header_row`** — scans the first several rows of a sheet for the row
that actually looks like column headers (rather than a title/logo row), since
raw exports put 3–4 decorative rows above the real header.

**`load_workbook_sheets`** — opens every `.xlsx` in a folder, loads every
sheet whose header can be detected, skips the ones that can't (logged as
warnings — these are reference/cover sheets, not data loss), and tags every
row with which file/sheet it came from (`source_file`, `REPORT_TYPE` for
claims / `BENEF_SHEET` for beneficiaries).

**Result:**
- Claims: 1,017,993 rows combined across all 4 workbooks' `Data` / `Recovery`
  / `PendingAuthorizations` sheets.
- Beneficiaries: 159,981 rows combined across all 5 workbooks.

---

## 3. Cleaning & standardization (`notebook §5`)

- **Join key.** `BENEFICIARYIDENTIFIER` (claims side) and `BENEFICIARYID`
  (beneficiary side) look like a matching ID pair but are two *different*
  identifier systems (a short hex code vs. a sequential numeric ID) that
  never overlap — joining on them silently matches 0 rows. The reliable
  shared key is **beneficiary full name + date of birth**, built the same way
  on both sides.
- **Dates** parsed to real datetimes.
- **Financial columns** (`PAYERAMOUNTCLAIMED`, `BENEFICIARYAMOUNTCLAIMED`,
  `TOTALAMOUNTCLAIMED`, `TOTALPAYABLE`, `PAYERSHARE`,
  `BENEFICIARYSHAREAPPROVED`, `DEDUCTIONAMOUNT`, …) coerced to numeric, with
  a `_WAS_MISSING` flag recorded **before** filling blanks with 0.0 — this
  keeps a genuine $0 claim distinguishable from a blank/unparseable source
  cell, rather than silently conflating the two.

---

## 4. Business logic (`notebook §6`)

**A. Status definitions.** Each `STATUS` code mapped to a plain-English
definition (17 statuses, from `Initial`/`Authorized` through
`Settled`/`Rejected`/`Void`).

**B. Rejected/Void split.** These are split into their own frame
(`df_excluded_claims`, exported separately) rather than silently dropped —
they carry no recovery information and aren't valid training rows, but
they're kept on disk for audit purposes. *(In this data pull, 0 rows were
Rejected/Void.)*

**C. Beneficiary merge.** Left-join on name+DOB matched **944,205 / 1,017,993**
claims (92.8%) to a beneficiary record. 16 duplicate `_beneficiary`-suffixed
columns dropped (same fact already present on the claims side). 3 sparse
columns dropped at ≥97% missing: `REINSURER` (only one source file ever
populates it — a real gap in the source data, not a bug), `Recovery Notes`,
`RecoveryNotes`.

**D. Post-termination claims deleted.** 9 rows where the service date fell
after the policy's expiry date were **removed**, not just flagged — a claim
serviced outside the covered period isn't a valid case for a recovery model
to learn from. (Exported separately to `excluded_post_termination_claims.csv`
for audit.)

**E. Missing-core-field rows deleted.** Rows missing any of
`TOTALAMOUNTCLAIMED`, `TXNDATE`, `STATUS`, `PROVIDERTYPE`, `CLAIMID`,
`ICD CHAPTER` were dropped — **178,775 rows (17.6% of the pre-cleaning
total)**. This is the single largest reduction in the pipeline; these are
unusable for modeling regardless of any imputation strategy.

**F. Recovery sub-table.** Rows with `RECOVERY/EX-GRATIA > 0` pulled into
their own frame (`df_recovery`) — this becomes the basis of the
`RECOVERY_FLAG` label in Phase 3.

### Cleaning funnel

| Stage | Rows removed | Cumulative rows remaining |
|---|---|---|
| Combined raw claims | – | 1,017,993 |
| Rejected/Void | 0 | 1,017,993 |
| Post-termination | 9 | 1,017,984 |
| Missing core field | 178,775 | 839,209 |

**Final cleaned dataset (`df_final`): 839,209 rows, 79 columns.**
Recovery/ex-gratia rows: **614 (0.073% of claims)** — the class-imbalance
number that drives every modeling decision in Phase 3.

---

## 5. Exports (`notebook §7`)

Written to `data/processed/`:

| File | Content |
|---|---|
| `consolidated_claims_all_reports.csv` | `df_final` — 839,209 cleaned claims, all pre- and post-adjudication columns, raw (unscaled) values |
| `recovery_ex_gratia_claims.csv` | The 614 recovered/ex-gratia claims, with recovery-specific columns |
| `excluded_rejected_void_claims.csv` | Rejected/Void claims (0 rows this pull) |
| `excluded_post_termination_claims.csv` | The 9 post-termination rows removed in step D |
| `claims_ml_ready.csv` | The Phase 3 modeling export — see §7 below |

---

## 6. Exploratory analysis (`notebook §8`)

Every chart in this section is built around the target variable — comparing
recovered claims against everything else — so the patterns map directly onto
candidate model features rather than being generic volume/demographic bar
charts. Key findings that carry into modeling:

- **Class imbalance is severe**: 614 positives out of 839,209 claims (0.073%).
  A plain-accuracy classifier predicting "never recoverable" would score
  99.93% while catching zero real recoveries — hence the entire Phase 3
  emphasis on PR-AUC and rank-based (precision/recall@k%) metrics instead of
  accuracy. 
- **Recovery rate by diagnosis (ICD chapter, min. 300 claims):** highest rates
  are in low-volume, clinically unusual chapters — *Mental, Behavioral and
  Neurodevelopmental disorders* (0.95%, n=634), *Congenital malformations*
  (0.54%, n=742), *Certain perinatal conditions* (0.43%, n=467) — followed by
  higher-volume chapters like *Diseases of the digestive system* (0.22%,
  n=101,051). This is exactly the kind of pre-adjudication signal
  (`ICD CHAPTER`) that should show up in feature importance later.
- **Recovery rate by service category (`FOB`):** *Optical* (1.92%),
  *Psychiatry* (1.79%), and *Wellness* (1.06%) run several times higher than
  *Out-Patient* (0.066%), the largest category by volume (773,222 claims).
- **Recovery rate by network type:** *Super-Restricted Network* (0.111%) and
  *Comprehensive Network* (0.105%) run roughly 2.5× the rate of *General
  Network* (0.045%) — both known at submission time, both legitimate
  pre-adjudication risk cuts.
- **Recovery rate by provider type:** *Hospital* claims carry both the
  highest recovery rate (0.104%) and by far the largest recovered-amount
  total (AED 499,280 across 328 recovered claims, vs. AED 64,327 for
  Polyclinic/Diagnostic Center and AED 52,194 for Pharmacy) — this directly
  supports the BI objective of ranking providers by recoverable-claim
  exposure.
- **Claim amount / submission delay:** recovered claims are checked against
  non-recovered ones for systematic differences in claimed amount and
  service-to-reception delay — both become engineered features
  (`CLAIM_AMOUNT_LOG`, `DAYS_TO_RECEPTION`) rather than being left as raw
  exploratory observations.

---

## 7. Building the ML-ready export (`notebook §9`)

`df_final` is a clean, analyst-friendly dataset, but it mixes **pre-adjudication**
facts (known at submission: patient profile, diagnosis, provider, claim
amount, dates) with **post-adjudication** facts (only known once a claim has
already been processed/settled/recovered). A model meant to score claims
*before payment* must never see post-adjudication columns as inputs.

### 7.1 Label

```python
RECOVERY_FLAG = (RECOVERY/EX-GRATIA > 0).astype(int)
```

### 7.2 Leakage columns excluded (14)

`STATUS`, `STATUS_DESCRIPTION`, `TOTALPAYABLE`, `PAYERSHARE`,
`BENEFICIARYSHAREAPPROVED`, `DEDUCTIONAMOUNT`, `RECOVERY/EX-GRATIA`,
`RECOVERYTYPE`, `PO`, `ORDERDATE`, `CHEQUENBR`, `SETTLEDATE`, `ASSESSMENT`,
`PAYABLE NON IP CONSULTATION` — every one of these is only known after a
claim has already been adjudicated/settled, so including any of them would
let the model "cheat" by learning the answer instead of the risk pattern.

### 7.3 Identifier / metadata columns excluded (21)

`CLAIMID`, `AUTHORIZATIONNBR`, `INVOICENBR`, `BENEFICIARYID`,
`BENEFICIARYIDENTIFIER`, `MASTERCONTRACT`, `CONTRACT`, `PRINCIPAL`,
`BENEFICIARY`, `PAYER`, `POLICYNO`, `POLICYNUMBER`, `CARDNUMBER`,
`NATIONALIDENTITYNO`, `ENDORSEMENTID`, `ENDOTYPE`, `ENDORSEMENTDATE`,
`ISSUEDATE`, `REPORT_TYPE`, `source_file`, `BENEF_SHEET` — free text,
administrative metadata, or row-unique identifiers that can't generalize as
model inputs. (This is also why Phase 3 cannot cleanly join model output back
to a raw AED recovery figure — see the Phase 3 doc's Step 12 discussion.)

### 7.4 Feature engineering — 10 new columns

| Feature | What it captures | Why it matters for recovery risk |
|---|---|---|
| `DAYS_TO_RECEPTION` | Gap between service date and claim reception | Unusual submission lag (too long or suspiciously immediate) is a standard claims-investigation red flag |
| `POLICY_DAYS_REMAINING` | Days left before policy expiry at claim time | "Use it before you lose it" utilization spikes near policy lapse are a known pattern |
| `MEMBER_TENURE_DAYS` | Days enrolled before filing | A large claim filed shortly after enrollment is a classic non-disclosure / opportunistic-enrollment pattern |
| `CLAIM_MONTH`, `CLAIM_QUARTER`, `CLAIM_IS_WEEKEND` | Seasonality/timing | Separates real seasonal illness patterns from anomalous timing (e.g. clustering around thin-staffing periods) |
| `CLAIM_AMOUNT_LOG` | Log of `TOTALAMOUNTCLAIMED` | Model hygiene, not a new signal — reshapes a heavily right-skewed amount so linear/distance-based models aren't dominated by a handful of huge claims |
| `RESUBMISSION_FLAG` | Whether the claim was resubmitted | A claim that needed correction and resubmission already had something wrong with it once — a standard fraud-literature red flag |
| `PROVIDER_CLAIM_FREQ`, `NETWORK_CLAIM_FREQ` | How often that provider/network appears in the data | Proxy for "established, high-volume provider" vs. "rarely-seen clinic" without one-hot-encoding thousands of individual provider identities; also feeds the BI provider-risk ranking objective |

### 7.5 Peer-deviation features (unsupervised outlier score)

Built on groups: `ICD CHAPTER` × `AGEBAND` × `FOB` (diagnosis, age band,
service category). For each claim, its peer group's median/mean/std claim
amount is computed, then:

- `PEER_ZSCORE` = how many standard deviations this claim's amount sits from
  its peer group's mean
- `AMOUNT_OVER_MEDIAN` = this claim's amount relative to its peer group's
  median

This is a lightweight, tariff-schedule-free version of "medically-justified
cost" — a claim priced far outside what similar claims for similar patients
typically cost is inherently suspicious, independent of any single feature.
These become both standalone anomaly scores (used as the Phase 3 baseline,
Step 4) and input features to the supervised models.

### 7.6 Encoding

- **18 low-cardinality columns** one-hot encoded (`GENDER`, `PROVIDERTYPE`,
  `FOB`, `CHRONIC`, `CLAIM CATEGORY`, `NETWORK TYPE`, `PROVIDER EMIRATE`, …)
  → ~75 dummy columns. 1 zero-variance dummy dropped (`CURRENCY_Arab Emirates
  Dirham` — every claim is in the same currency, so it carries no
  information).
- **10 high-cardinality columns** frequency-encoded instead of one-hot
  (`PROVIDER`, `NATIONALITY`, `ICD CHAPTER`, `ICD SUB CHAPTER`, `NETWORK`,
  `EMPNO`, `CATEGORY`, `PROVIDERCOUNTRY`, `PROVIDER GROUP`, `PRODUCT`) —
  one-hotting something like individual provider names would explode into
  thousands of columns and just memorize identities instead of learning
  behavior.
  > **Known limitation, stated in the notebook itself:** frequency encoding
  > is computed over the *whole* dataset here, not fit on the training fold
  > only. In a strict train/test regime this leaks a small amount of
  > test-period information into the encoding. Worth naming in the thesis
  > methodology section as a simplification, not silently ignoring.
- **9 raw datetime columns** dropped (`EFFECTIVEDATE`, `EXPIRYDATE`, `DOB`,
  `TXNDATE`, `DISCHARGEDATE`, `MEMBER START DATE`, `RECEPTIONDATE`,
  `CONTRACTEFFDATE`, `CONTRACTEXPDATE`) — their signal is already captured in
  the engineered `*_DAYS`/`CLAIM_MONTH` features; raw datetimes aren't usable
  model inputs directly.

### 7.7 Scaling

33 continuous numeric columns standardized (`StandardScaler`: mean 0, std 1)
— includes `AGE`, `LOS`, all claimed-amount columns, all engineered day-gap
features, and all frequency/peer-deviation features. One-hot dummies and
0/1 flags are left unscaled (already comparable range; standardizing adds no
value and roughly doubles `StandardScaler`'s memory footprint for ~80 dummy
columns across 839k rows for no benefit).

**`CLAIM_YEAR` is deliberately kept raw** (not scaled) — Phase 3's
train/valid/test split is built directly on this column, and a standardized
z-score would destroy the actual year values (2013/2014/2015/2016) needed for
that split.

### 7.8 Final export

```
data/processed/claims_ml_ready.csv
839,209 rows x 110 columns (109 features + RECOVERY_FLAG)
Positive class (RECOVERY_FLAG=1): 614 (0.073%)
```

No claim ID, no raw (unscaled) currency amount survives into this file by
design — both are excluded as identifier/leakage-adjacent columns. This has
one direct downstream consequence, addressed explicitly in Phase 3: an
AED-based business metric can't be computed from `claims_ml_ready.csv` alone
and has to be recovered by a **row-position join** back to
`consolidated_claims_all_reports.csv` (verified safe because both files share
the same row order and row count with zero misalignment — see the Phase 3
doc, Step 12).

---

## 8. Known data-quality notes (not pipeline bugs)

- **Beneficiary join keys don't match.** `BENEFICIARYIDENTIFIER` and
  `BENEFICIARYID` are different ID systems; fixed by joining on name+DOB
  instead (§3 above).
- **`REINSURER` is genuinely blank** in the source workbooks — only one file
  ever populates it. Dropped as a sparse column, not treated as a bug to fix.
- **Weak-label caveat** (carries into every Phase 3 result): "not recovered"
  doesn't necessarily mean "clean" — it may just mean nobody caught it yet.
  This is a standard, well-known limitation in fraud-detection literature
  under noisy/weak labeling, not a flaw specific to this pipeline's design.
