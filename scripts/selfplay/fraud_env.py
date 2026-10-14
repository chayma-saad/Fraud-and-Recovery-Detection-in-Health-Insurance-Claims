"""Gymnasium environment: the fraudster agent vs. a static XGBoost defender.

Episode = one baseline (synthetic) claim template. The agent has a "fraud
budget" (a multiple of the claim's legitimate amount) it's trying to extract
from the insurer, and up to MAX_STEPS submissions to do it in. Each step it
chooses what fraction of its *remaining* budget to submit as this step's
claimed amount; that modified claim is scored by the frozen defender.

- Not caught: the claimed amount is "kept" (reward), remaining budget shrinks,
  episode continues.
- Caught (defender's flagged probability exceeds the detection threshold):
  episode ends immediately, the rest of the budget is forfeited, and a
  penalty is applied — modeling that an audited claim also burns the
  remaining opportunity, not just this one submission.

This directly operationalizes the two example tricks named in the project
proposal: submitting several smaller claims instead of one big one
("splitting"), and picking amounts that don't cross the defender's alert
threshold ("staying under the radar") — both are outcomes of how the agent
learns to spend its per-step amount fraction, not hand-coded behaviors.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from xgboost import XGBClassifier

from common import (
    LABEL_COL, encode_rows, encode_amount_only, AMOUNT_IDX, LOG_AMOUNT_IDX,
)

MAX_STEPS = 5
FRAUD_BUDGET_MULTIPLIER = 3.0  # the agent is trying to extract up to 3x the claim's legitimate amount
CAUGHT_PENALTY_SCALE = 1.5  # penalty on getting caught, relative to the forfeited remaining budget


class ClaimFraudEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, synthetic_df, meta, defender: XGBClassifier, threshold: float, seed=None):
        super().__init__()
        self.synthetic_df = synthetic_df.reset_index(drop=True)
        self.meta = meta
        self.defender = defender
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)

        # Pre-encode every synthetic baseline claim once (amount fields get
        # overwritten per-step; everything else is frozen for the episode).
        self._encoded_pool = encode_rows(self.synthetic_df, meta)
        self.n_features = self._encoded_pool.shape[1]

        # Observation: encoded baseline claim (context) + remaining budget
        # fraction + steps remaining fraction.
        obs_dim = self.n_features + 2
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        # Action: what fraction of the REMAINING budget to submit this step,
        # as a symmetric [-1, 1] action (SB3/PPO convention) rescaled to
        # [0, 1] inside step() -- an unbounded/asymmetric Box around a
        # zero-mean Gaussian policy destabilizes early training.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._claim_idx = None
        self._base_amount = None
        self._budget_total = None
        self._budget_remaining = None
        self._step_count = None
        self._base_encoded = None

    def _get_obs(self):
        obs = self._base_encoded.copy()
        extra = np.array([
            self._budget_remaining / self._budget_total,
            (MAX_STEPS - self._step_count) / MAX_STEPS,
        ], dtype=np.float32)
        return np.concatenate([obs, extra]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._claim_idx = int(self._rng.integers(0, len(self.synthetic_df)))
        self._base_amount = max(float(self.synthetic_df.iloc[self._claim_idx]["TOTALAMOUNTCLAIMED"]), 1.0)
        self._budget_total = self._base_amount * FRAUD_BUDGET_MULTIPLIER
        self._budget_remaining = self._budget_total
        self._amount_kept = 0.0
        self._step_count = 0
        self._base_encoded = self._encoded_pool[self._claim_idx]
        return self._get_obs(), {}

    def step(self, action):
        frac = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] -> [0,1]
        claimed_amount = frac * self._budget_remaining
        claimed_amount = max(claimed_amount, 0.0)

        row = encode_amount_only(self._base_encoded, claimed_amount, self.meta,
                                  amount_idx=AMOUNT_IDX, log_idx=LOG_AMOUNT_IDX)
        detect_proba = float(self.defender.predict_proba(row.reshape(1, -1))[0, 1])
        caught = detect_proba >= self.threshold

        self._step_count += 1
        terminated = False
        info = {"claimed_amount": claimed_amount, "detect_proba": detect_proba, "caught": caught}

        if caught:
            # The flagged sub-claim itself is also seized on top of the
            # forfeited remaining budget -- getting caught loses everything,
            # not just what hadn't been submitted yet.
            forfeited = self._budget_remaining
            reward = -CAUGHT_PENALTY_SCALE * forfeited / self._budget_total
            terminated = True
            self._budget_remaining = 0.0
        else:
            reward = claimed_amount / self._budget_total
            self._budget_remaining -= claimed_amount
            self._amount_kept += claimed_amount

        truncated = (not terminated) and (
            self._step_count >= MAX_STEPS or self._budget_remaining <= 1e-6
        )
        info["budget_remaining"] = self._budget_remaining
        info["amount_kept"] = self._amount_kept
        info["kept_fraction"] = self._amount_kept / self._budget_total
        return self._get_obs(), reward, terminated, truncated, info
