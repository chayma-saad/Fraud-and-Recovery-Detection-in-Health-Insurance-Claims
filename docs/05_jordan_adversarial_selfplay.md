# Phase 4b — Jordan Dataset: Adversarial Self-Play

Source notebook: `notebooks/05_jordan_adversarial_selfplay.ipynb`
Generator script: `scripts/build_jordan_selfplay_notebook.py`
Pipeline scripts: `scripts/jordan_selfplay/{common,train_defender,train_ctgan,fraud_env,train_ppo,retrain_defender_v2}.py`

The Jordan-dataset equivalent of Phase 4, run **alongside** the Al Hilal
analysis (not replacing it), once `data/processed/jordan_data/` became
available mid-project. Same MVP scope agreed for Al Hilal: one PPO
fraudster vs. a static defender, plus one bonus round of defender
retraining — not the full multi-round co-evolution loop.

---

## What's different here, stated up front

Jordan's `defender_v1` is built on a leaner, fully self-authored feature
set (27 encoded dimensions vs. Al Hilal's 81) drawn directly from
`jordan_ml_ready.csv`'s own 34-column export — and it starts out
**weaker**: PR-AUC 0.0168 vs. Al Hilal's 0.0339. This isn't a bug; it's a
direct, expected consequence of Jordan's leakage-safe feature set being
smaller to begin with (Objective 1's own model, using the full 34 safe
columns, only reaches PR-AUC 0.0132 — see `docs/04_jordan_modeling.md`).
Everything downstream in this exercise reads differently as a result, and
this document reports what actually happened rather than assuming it
would mirror Al Hilal's numbers.

---

## Step 1 — Synthetic data (CTGAN)

Trained on a 40,000-row sample of real 2013–2015 claims (oversampling the
281 real collusion-flagged claims), conditioned on `COLLUSION_FLAG`.

**One deliberate deviation from Al Hilal's approach:** `icd9code` (2,023
distinct raw codes) was **not** included as a CTGAN discrete training
column — that cardinality would blow up CTGAN's internal one-hot
transformer for no real benefit. Diagnosis code is instead
bootstrap-sampled from the real distribution jointly with the other
amount-independent columns (timing, patient/provider history) when a
synthetic claim is built into a baseline template — same device, just
applied to one more column than Al Hilal needed to.

```
CTGAN(epochs=150) training took 448s (~7.5 minutes) -- notably faster than
Al Hilal's ~21 minutes, because Jordan's discrete column set is much
smaller (5 low-cardinality categoricals vs. Al Hilal's 12, including one
20-category ICD chapter column).
```

**Data quality notes:**
- **479 rows (2.4%)** came out with a non-positive `claimedvalue` —
  clipped to a floor of 0.1, comparable to Al Hilal's 2.9% clip rate.
- **Synthetic `COLLUSION_FLAG` rate: 8.3%**, vastly above the real training
  rate (0.079%) — the same CTGAN training-by-sampling artifact documented
  for Al Hilal. Doesn't affect the environment, which never reads this
  column as a feature.

---

## Step 2 — The static defender (`defender_v1`)

```
XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
              scale_pos_weight=negative/positive, tree_method="hist")
```

Trained on the 27-dimension encoding (Objective 1's safe feature set minus
`PEER_ZSCORE`/`AMOUNT_OVER_MEDIAN`, excluded for the same reason Al Hilal's
were: they're `claimedvalue`-dependent and this MVP doesn't dynamically
recompute peer-group statistics per step).

```
Defender v1 holdout (2016-2017) PR-AUC:  0.0168
Defender v1 holdout ROC-AUC:             0.8381
Defender v1 Recall@5%:                   39.5%
Detection threshold (95th pct of train scores): 0.0624
```

For comparison, Objective 1's own full-feature LightGBM model (34 safe
columns, including the peer-deviation pair) reaches PR-AUC 0.0132 on the
narrower 2017-only test set — the two numbers aren't directly comparable
(different test windows, different feature counts) but confirm the same
order of magnitude: Jordan's collusion signal is real but harder to
separate than Al Hilal's recovery signal, on both models built for it.

---

## Step 3 — The environment

Identical mechanics to Al Hilal's `ClaimFraudEnv` (see
`docs/03_adversarial_selfplay.md` Step 4 for the full description): a
fraud budget of 3× the claim's legitimate `claimedvalue`, up to 5
submissions, everything forfeited on capture. Only the underlying encoder
differs — `encode_amount_only` here recomputes just the `claimedvalue`
entry (Jordan's export has no separate log-amount column the way Al
Hilal's `CLAIM_AMOUNT_LOG` did).

---

## Step 4 — Baseline: random policy vs. `defender_v1`

```
Random policy attack success rate: 93.3%
Random policy mean budget kept:    91.2%
```

**Even weaker than Al Hilal's already-weak defender** (86.0% random-policy
success there). This tracks directly with `defender_v1`'s own PR-AUC
(0.0168 vs. Al Hilal's 0.0339): a classifier with less separating power has
a wider blind spot, and blind amount inflation walks through it almost
every time.

---

## Step 5 — PPO training vs. `defender_v1`

```
PPO("MlpPolicy", n_steps=1024, batch_size=256, learning_rate=3e-4)
Trained for 200,000 timesteps.
```

```
Random policy:  attack success rate 93.3%, mean budget kept 91.2%
Trained policy: attack success rate 94.0%, mean budget kept 94.0%
```

**Even less room to improve than Al Hilal's already-modest gain** (86.0% →
90.3% there). The training curve fluctuates between roughly 92% and 97%
for the entire 200,000 timesteps with no clear upward trend — consistent
with there being essentially no learnable ceiling left above random chance
against a defender this weak. This is a clean, if unglamorous,
confirmation of Al Hilal's Step 5 finding, taken to its logical extreme:
**a weak enough defender is beaten by almost anything, learned or not.**

---

## Step 6 — What strategy did the agent learn?

Inspecting all 282 successful evasive sub-claims:

| Sub-claims per episode | Episodes |
|---|---|
| 1 (single-shot) | 282 (100%) |

**Result: zero splitting observed at all** — more extreme than Al Hilal's
90%/10% split. This is the logical extreme of the Step 4/5 finding: when a
single full-budget claim already evades detection essentially every time,
there is no scenario where salami-slicing would help. This is itself an
informative negative result: **splitting-based evasion is a response to a
defender with at least some amount sensitivity, not an inherent RL-agent
behavior.** Al Hilal's comparatively stronger `defender_v1` was what
elicited the occasional split there; Jordan's weaker one never requires
it.

---

## Step 7 — Round 2: retraining the defender

Same method as Al Hilal: the 282 successful evasive sub-claims are labeled
as collusion-positive, folded into the real training data (281 original
positives + 282 adversarial = 563), and the defender is retrained. The
**frozen** trained agent — not retrained itself — is re-evaluated against
the result.

```
Defender v2 holdout PR-AUC:  0.0217  (v1: 0.0168)
Defender v2 holdout ROC-AUC: 0.8281  (v1: 0.8381)
Defender v2 detection threshold: 0.5511  (v1: 0.0624)

Same frozen agent vs. defender_v1: 94.0% attack success
Same frozen agent vs. defender_v2: 68.7% attack success, 68.7% mean kept
```

**Result: a real, substantial drop — 94.0% → 68.7% — but a smaller one
than Al Hilal's 90.3% → 22.7%.** Two honest reasons, not one:

1. **`defender_v1` started from a much weaker baseline.** There was more
   ground to cover with the same single round of retraining, and one round
   evidently isn't enough to close a gap this wide.
2. **Proportionally, more was added here** (282 adversarial examples
   against 281 original positives — roughly doubling the positive-label
   count) **than for Al Hilal** (337 against 252, a ~1.3× increase) — yet
   the result moved less in relative terms. That combination suggests
   Jordan's defender needs either more retraining rounds or a richer safe
   feature set (starting with the excluded peer-deviation pair) to close
   the gap the way Al Hilal's did in a single step.

**Unlike Al Hilal, `defender_v2`'s real-holdout PR-AUC also genuinely
improved** (0.0168 → 0.0217, a proportionally large jump) while ROC-AUC
held roughly steady (0.838 → 0.828) — a cleaner result than Al Hilal's
case, where PR-AUC improved only slightly and ROC-AUC gave back some
ground. Here, the retraining round didn't just patch the one discovered
hole narrowly — it measurably generalized better on real, unseen data too.

---

## Step 8 — Red-team summary

| Finding | Detail |
|---|---|
| Undetected amount inflation is the *only* observed strategy | Random policy already wins 93.3% of the time; the trained agent found zero use for splitting across all 282 successful episodes |
| A weaker starting defender is beaten even more thoroughly | `defender_v1`'s PR-AUC (0.0168) is about half Al Hilal's (0.0339), and its random-baseline attack-success rate is correspondingly higher (93.3% vs. 86.0%) |
| One retraining round still helps, just doesn't fully close the gap in one step | 94.0% → 68.7% attack success, with `defender_v2`'s real-data PR-AUC genuinely improving (unlike Al Hilal's mixed PR-AUC/ROC-AUC result) |
| Suggested patch | Same lever as Objective 1's own recommendation: add back `PEER_ZSCORE`/`AMOUNT_OVER_MEDIAN` with dynamic per-step recomputation — this MVP's defender excludes them for tractability, and that exclusion is the most direct explanation for how thoroughly amount inflation succeeds here |

**Headline:** a second, independent dataset — a different country, a
different fraud concept (investigator-flagged collusion vs. post-payment
recovery), a different feature set — shows the same core pattern Al Hilal
did: a fraud defender missing peer-group context is beaten by simple
amount inflation almost every time, learned or random, and adversarial
retraining measurably narrows that gap even when — as it does here — one
round isn't enough to close it outright.

---

## What was explicitly out of scope (same as Al Hilal)

- Full multi-round co-evolution loop — only one retraining round was run.
- PettingZoo multi-agent setup — single-agent Gymnasium environment.
- LLM-based fraudster / tool-calling agent.
- Peer-deviation-aware defender inside the RL simulator — and here, more
  than for Al Hilal, this looks like the single highest-value next step:
  Jordan's defender needed *more* than one round to make comparable
  progress, and the excluded peer features are the most likely reason why.
