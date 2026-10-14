# Phase 4 (Idea 2 MVP) — Adversarial Self-Play Red-Teaming: Step-by-Step Documentation

Source notebook: `notebooks/03_adversarial_selfplay.ipynb`
Generator script: `scripts/build_selfplay_notebook.py`
Pipeline scripts: `scripts/selfplay/{common,train_defender,train_ctgan,fraud_env,train_ppo,retrain_defender_v2}.py`

**Core question.** Phase 3 only learns from fraud that already happened.
Can an AI *invent* new fraud strategies against our own defenses — so the
holes get patched before a real person finds them?

## Scope decision (agreed before building)

The full proposal (Idea 2) describes a PettingZoo multi-agent environment,
an attacker that keeps adapting across thousands of rounds, and a defender
that co-evolves with it round by round. That is closer to its own thesis
chapter than a notebook add-on. Before building, three scope questions were
resolved explicitly with the project owner:

1. **Build an MVP first**: a single PPO fraudster agent (Gymnasium, not
   PettingZoo) against one **static** defender — not the full co-evolution
   loop — to prove the core mechanism works end to end before investing in
   the harder multi-agent, many-round version.
2. **Real data boundary**: CTGAN trains on real claims; everything
   downstream in the RL sandbox only ever sees CTGAN-sampled synthetic
   claims — no real patient data reaches the simulator.
3. **New dependencies**: `torch`, `gymnasium`, `stable-baselines3`, `ctgan`
   installed into `scripts/venv` (CPU-only torch build — the available GPU's
   driver is old enough, CUDA 11.2, that CPU was simpler and plenty fast for
   this model size).

This document covers exactly that MVP, plus one bonus round of defender
retraining that goes a step beyond the agreed MVP scope (see Step 7) because
it turned out to be cheap and gave a much more informative result than
stopping at the static-defender comparison alone.

---

## Step 1 — Feature set and encoder (`scripts/selfplay/common.py`)

**Why not reuse Phase 3's 109-column pipeline:** the RL environment needs to
cheaply re-score a claim every time the agent changes its claimed amount —
potentially hundreds of thousands of times across training. That requires
knowing exactly how every input feature was constructed, so the
amount-dependent ones can be recomputed per step without re-deriving the
whole encoding pipeline. Phase 3's frequency encodings and peer/provider
aggregate statistics were fit dataset-wide inside that notebook and aren't
reproducible at that granularity here.

Instead, a smaller, fully self-contained feature set was built:

- **4 continuous columns**: `AGE`, `TOTALAMOUNTCLAIMED`, `NETPREMIUM`,
  `GROSSPREMIUM`, plus a derived `CLAIM_AMOUNT_LOG` — standardized with
  scaling parameters fit once on the real training data (`≤2014`) and frozen.
- **12 categorical columns**, one-hot encoded: `GENDER`, `MARITAL STATUS`,
  `AGEBAND`, `DEPENDENCY`, `NETWORK TYPE`, `FOB`, `CHRONIC`, `PREEXISTING`,
  `CLAIMTYPE`, `PROVIDER EMIRATE`, `PROVIDERTYPE`, `ICD CHAPTER` — real gaps
  filled the same way as Phase 2 (`PROVIDER EMIRATE` NaN → `"Unknown"`;
  `NETPREMIUM`/`GROSSPREMIUM` NaN → median).
- **8 amount-independent engineered columns** pulled pre-scaled directly
  from `claims_ml_ready.csv` (row-aligned, same verified-safe join as Phase
  3's Step 12): `MEMBER_TENURE_DAYS`, `DAYS_TO_RECEPTION`,
  `RESUBMISSION_FLAG`, `PROVIDER_CLAIM_FREQ`, `NETWORK_CLAIM_FREQ`,
  `CLAIM_MONTH`, `CLAIM_QUARTER`, `CLAIM_IS_WEEKEND`. These are safe to
  freeze per baseline claim because the agent never controls timing,
  provider identity, or resubmission in this MVP — only claimed amount.

Total encoded width: **81 dimensions**. `encode_amount_only()` is the key
function the environment calls every step: given an already-encoded baseline
row, it recomputes *only* the two amount-derived entries
(`TOTALAMOUNTCLAIMED_scaled`, `CLAIM_AMOUNT_LOG_scaled`) for a new claimed
amount, leaving demographics/provider/diagnosis/timing frozen — this is what
makes per-step re-scoring cheap enough to run at RL scale.

**Deliberately excluded**: `PEER_ZSCORE`, `AMOUNT_OVER_MEDIAN`,
`PROVIDER_AVG_AMOUNT` — Phase 3's strongest amount-derived features. They
depend on peer/provider group statistics that would themselves need to
shift as the agent's claimed amount changes, which this MVP's per-step
recompute function doesn't handle. This exclusion is the direct cause of
the Step 4 finding below — stated here so it isn't a surprise later.

---

## Step 2 — `defender_v1`: the static XGBoost defender

```
XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
              scale_pos_weight=negative/positive, tree_method="hist")
```

Trained on the 81-dimension encoding of real claims, years ≤2014; evaluated
on the 2015-2016 holdout, same temporal-split discipline as Phase 3:

```
Defender v1 holdout PR-AUC:      0.0339
Defender v1 holdout ROC-AUC:     0.8318
Defender v1 Recall@5%:           55.5%
Detection threshold (95th pct of train scores): 0.0117
```

**This is intentionally much weaker than Phase 3's XGBoost (PR-AUC 0.506)**
— the reduced feature set (Step 1) is missing exactly the features Phase 3
found most useful for spotting amount-based anomalies. This was checked
directly during development: a first attempt using only the raw 18 columns
(no timing/provider-history features) scored PR-AUC 0.0040; adding the 8
amount-independent engineered columns from Phase 3 brought it up to 0.0339
— confirming those features carry real signal even in this reduced form,
and that the gap to Phase 3's full model is attributable to the excluded
peer-deviation/provider-amount features specifically, not a broken pipeline.

The **detection threshold** (0.0117) is the 95th percentile of the
defender's own scores on its training data — i.e., a claim is "caught" if
it would land in the riskiest 5% of claims by this defender's own judgment,
mirroring the audit-queue framing from Phase 3.

---

## Step 3 — CTGAN synthetic claims

```
CTGAN(epochs=150), trained on a 40,000-row sample of real 2014-and-earlier
claims (oversampling the 252 real recovered claims so the generator sees
enough of them), conditioned on RECOVERY_FLAG.
Training took ~1,240s (~21 minutes) on CPU.
```

20,000 synthetic claims sampled from the trained generator.

**Data quality notes, checked and documented rather than assumed:**
- **581 rows (2.90%)** came out with a non-positive `TOTALAMOUNTCLAIMED` —
  clipped to a floor of 1.0 rather than resampled or silently left negative.
  Normal behavior for a GAN modeling a heavily right-skewed continuous
  column; the clip rate itself is a reasonable data-quality check on the
  generator, not something to hide.
- **Synthetic `RECOVERY_FLAG` rate: 6.48%**, vastly higher than the real
  training rate (0.084%). This is CTGAN's documented "training-by-sampling"
  method (Xu et al., 2019): during training it deliberately oversamples rare
  discrete-column values via a log-frequency sampler so the generator learns
  their shape, and unconditioned `.sample()` calls reproduce that same
  skewed distribution rather than the true marginal. **This does not affect
  anything downstream** — `RECOVERY_FLAG` from the synthetic pool is never
  read by `ClaimFraudEnv` as a feature; it's carried-through metadata only,
  present because CTGAN needed it as a conditioning column during training.

---

## Step 4 — `ClaimFraudEnv` (Gymnasium environment)

Each **episode** = one synthetic baseline claim, sampled fresh from the
20,000-row pool. The agent has a **fraud budget** of 3× the claim's
legitimate amount and up to **5 submissions** to extract it in.

- **Observation**: the baseline claim's 81-dim encoded feature vector, plus
  remaining-budget fraction and steps-remaining fraction (83 dims total).
- **Action**: a single continuous value in `[-1, 1]` (SB3/PPO convention,
  rescaled to `[0, 1]` inside the environment), interpreted as *what
  fraction of the remaining budget to submit this step*.
- **Step logic**: the chosen amount is written into a copy of the baseline
  row via `encode_amount_only()`, scored by `defender_v1`. If the predicted
  probability clears the detection threshold, the claim is **caught**: the
  episode ends immediately and the *entire remaining budget is forfeited*
  (not just the amount not yet submitted) — modeling that an audited claim
  burns the whole opportunity, not only that one submission. If not caught,
  the amount is banked, the budget shrinks, and the episode continues (up to
  5 steps, or until the budget is exhausted).

This directly operationalizes the two example tricks named in the proposal
— **splitting** a claim into several smaller ones, and **staying under an
alert threshold** — as *emergent outcomes* of the amount-fraction policy the
agent learns, not hand-coded behaviors the environment forces it to use.

**Bug found and fixed during development**: the environment's
`kept_fraction` metric was computed from `budget_remaining`, which the step
logic zeroes out on capture for internal bookkeeping — this made every
*caught* episode incorrectly report `kept_fraction = 1.0` (as if nothing was
lost), even though the whole point of getting caught is forfeiting the rest.
Fixed by tracking a separate `amount_kept` accumulator that only increases
on non-caught steps. **This bug never affected `attack_success_rate`**
(tracked independently via the `caught` flag) — only the budget-kept
metric, which is why the two training runs before and after the fix show
identical success-rate curves but different kept-fraction curves.

---

## Step 5 — PPO training vs. `defender_v1`

```
PPO("MlpPolicy", n_steps=1024, batch_size=256, learning_rate=3e-4)
Trained for 200,000 timesteps; attack success rate measured on 300 held-out
episodes every 10,000 timesteps.
```

```
Random policy:  attack success rate 86.0%, mean budget kept 84.6%
Trained policy: attack success rate 90.3%, mean budget kept 88.2%
```

**This is the most important honest finding in this MVP, and it needs to be
read correctly.** A *random* policy already wins 86% of the time against
`defender_v1`. This is not evidence that real-world insurance fraud is
trivially easy — it is a direct, traceable consequence of Step 1's decision
to exclude peer-deviation/provider-amount features from this defender:
without them, `defender_v1` has no way to judge whether a claimed amount is
anomalous *relative to what similar claims usually cost* — only whether the
raw claim looks unusual against the whole population's demographic/timing
mix. A single 3×-inflated claim doesn't look unusual on those terms alone.

The training curve (attack success rate vs. timesteps) is essentially flat
from the very first 10,000-step checkpoint onward — consistent with, not
contradictory to, the above: when a random policy already wins 86% of the
time, there is very little room left for learning to improve on. The
practical implication for the thesis: **a weak defender isn't just
vulnerable to a sophisticated learned attack, it's vulnerable to almost
anything**, which argues for prioritizing the known fix (peer-deviation
features, already built in Phase 3) over building a more elaborate attacker.

---

## Step 6 — What strategy did the agent learn?

Inspecting the 337 successful evasive sub-claims saved from evaluating the
final trained policy, grouped by episode split count:

| Sub-claims per episode | Episodes | Share |
|---|---|---|
| 1 (single-shot) | 244 | 90.0% |
| 2 | 10 | 3.7% |
| 3 | 5 | 1.8% |
| 4 | 2 | 0.7% |
| 5 | 10 | 3.7% |

**Result: the dominant strategy (90%) is a single, large, one-shot claim for
the full 3× fraud budget — not splitting.** This confirms the Step 5
diagnosis directly: because `defender_v1` can't distinguish an inflated
amount from a normal one without peer-group context, the agent doesn't need
to disguise its claim as several smaller, individually-unremarkable
submissions — brute-force amount inflation already works most of the time.
The minority (~10%) of episodes that do split (up to 5 sub-claims) are
presumably baseline claims with a riskier demographic/provider/timing
combination, where one large claim would be too conspicuous and
salami-slicing genuinely helps.

**Red-team takeaway, stated plainly**: undetected amount inflation is the
*primary* exposure this defender has; claim-splitting is a *secondary*
strategy the agent falls back on only when the primary one doesn't work.
A red-team report that only warned about splitting (matching the literal
example in the proposal) would have missed the bigger, more common hole.

---

## Step 7 — Round 2: retraining the defender (bonus, beyond agreed MVP scope)

The agreed MVP scope stopped at "single agent vs. static defender." This
step goes one round further because it was cheap to add and gives a far
more informative result than stopping at Step 5: it's a single-round taste
of the proposal's full co-evolution loop (Step 5 in the original pipeline
description — "feed the discovered attacks to the defender and retrain").

**Method**: the 337 successful evasive sub-claims from Step 6 are labeled as
fraud (1) and folded into the real training data (252 original positives +
337 adversarial positives = 589 positives on top of the same negatives).
XGBoost is retrained with the same hyperparameters as `defender_v1`,
producing `defender_v2`. Critically, **the PPO agent is not retrained** —
the exact same frozen policy from Step 5 is re-evaluated against the new
defender, isolating the effect of the defender's improvement.

```
Defender v2 holdout (2015-2016) PR-AUC:  0.0489  (v1: 0.0339)
Defender v2 holdout ROC-AUC:             0.7810  (v1: 0.8318)
Defender v2 detection threshold:         0.3030  (v1: 0.0117)

Same frozen agent vs. defender_v1: 90.3% attack success, 88.2% mean kept
Same frozen agent vs. defender_v2: 22.7% attack success, 21.1% mean kept
```

**Result: a dramatic drop — 90.3% → 22.7% attack success for the identical,
unmodified attacker — from retraining on just 337 discovered examples.**
This closely matches the kind of swing the project proposal frames as the
headline KPI ("a drop from 90% to 12% quantifies how much stronger the
system became"), achieved here with a single round rather than the full
proposed loop.

**Mechanism, stated honestly rather than oversold**: `defender_v2` did not
get meaningfully better at fraud detection in general — its real-data
holdout PR-AUC only moved from 0.034 to 0.049 (a real but modest gain), and
its ROC-AUC actually *dropped* slightly (0.832 → 0.781), suggesting the
retraining nudged the decision boundary toward amount ranges resembling the
agent's specific attacks at some small cost to broader generalization. What
happened is narrower and more mechanical: the defender specifically learned
"claimed amounts in the range this attacker was using are suspicious,"
which directly closes the one hole this one attacker had found. That is
both the promise and the honest limit of this method:

- **The promise**: discovered holes get closed fast and cheaply — one
  retraining round, 337 examples, built in minutes.
- **The limit**: a single round doesn't generalize to holes *not yet*
  discovered. This is precisely the argument for running many rounds with
  an attacker that keeps adapting (the full co-evolution loop), rather than
  treating one round as sufficient.

---

## Step 8 — Red-team report (business framing)

| Finding | Detail | Suggested patch |
|---|---|---|
| Undetected amount inflation | A single-shot claim at 3× the legitimate amount evades `defender_v1` in ~90% of successful attempts — the dominant discovered strategy | Add peer-group deviation features (`PEER_ZSCORE`, `AMOUNT_OVER_MEDIAN` — already built and validated in Phase 3) so amount is judged relative to similar claims, not in isolation |
| Budget splitting (minority strategy) | ~10% of successful episodes split the fraud budget across 2-5 sub-claims, likely for baseline claims where one large claim would be too conspicuous | Add resubmission/frequency-window features flagging multiple claims from the same profile within a short window (Phase 2's `RESUBMISSION_FLAG`/`PROVIDER_CLAIM_FREQ` are a starting point) |
| A weak defender is beaten by almost anything | Random policy already achieves 86% attack success — the gap between "undefended" and "RL-optimized attack" is small when the baseline defense has an amount blind spot | Prioritize closing the known blind spot over hardening against sophisticated attacks the current defender isn't even distinguishing from random noise |
| One retraining round closes a large, specific hole | 90.3% → 22.7% attack success after folding in 337 discovered examples; real-data PR-AUC only moved 0.034→0.049 (narrow, not general, improvement) | Repeat the round automatically, with an attacker that keeps adapting rather than staying frozen, to approach the full proposed co-evolution loop |

**Headline for the thesis**: a fraud detector missing peer-group context can
be beaten by simple amount inflation nearly 9 times out of 10 — even by
random chance, without any learning at all. A single round of adversarial
red-teaming, folding just 337 discovered evasive claims back into training,
cuts that down to roughly 2 times out of 10. Scaling this into the full
proposed co-evolution loop (many rounds, an attacker that keeps adapting,
PettingZoo multi-agent tooling for a true two-sided game) is the natural
next iteration, not a change of approach.

---

## What was explicitly out of scope for this MVP

- **Multi-round co-evolution loop.** Only one retraining round was run (Step
  7), as a proof of concept. The proposal's "thousands of rounds" framing
  and its round-by-round attack-success-rate KPI would need this loop
  automated and run to convergence.
- **PettingZoo multi-agent setup.** The environment is single-agent
  Gymnasium (attacker only, static defender as an external scoring
  function), not a true two-sided PettingZoo game where both agents act
  within the same environment step.
- **LLM-based fraudster / tool-calling agent.** Only the PPO (Stable-
  Baselines3) path from the proposal's model table was built.
- **Full peer-deviation-aware defender.** `defender_v1`/`v2` deliberately
  exclude the amount-derived features that would make them comparable in
  strength to Phase 3's model — see Step 1 and Step 5 for why, and Step 8
  for why closing this gap is the single highest-value next step.
