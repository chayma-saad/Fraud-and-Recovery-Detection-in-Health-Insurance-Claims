"""Gymnasium environment: the fraudster agent vs. a static XGBoost defender
-- Jordan-dataset equivalent of scripts/selfplay/fraud_env.py. Identical
mechanics (see that file's docstring for the full rationale); only the
encoder import differs.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from xgboost import XGBClassifier

from common import encode_rows, encode_amount_only, AMOUNT_IDX

MAX_STEPS = 5
FRAUD_BUDGET_MULTIPLIER = 3.0
CAUGHT_PENALTY_SCALE = 1.5


class ClaimFraudEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, synthetic_df, meta, defender: XGBClassifier, threshold: float, seed=None):
        super().__init__()
        self.synthetic_df = synthetic_df.reset_index(drop=True)
        self.meta = meta
        self.defender = defender
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)

        self._encoded_pool = encode_rows(self.synthetic_df, meta)
        self.n_features = self._encoded_pool.shape[1]

        obs_dim = self.n_features + 2
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._claim_idx = None
        self._base_amount = None
        self._budget_total = None
        self._budget_remaining = None
        self._amount_kept = None
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
        self._base_amount = max(float(self.synthetic_df.iloc[self._claim_idx]["claimedvalue"]), 0.1)
        self._budget_total = self._base_amount * FRAUD_BUDGET_MULTIPLIER
        self._budget_remaining = self._budget_total
        self._amount_kept = 0.0
        self._step_count = 0
        self._base_encoded = self._encoded_pool[self._claim_idx]
        return self._get_obs(), {}

    def step(self, action):
        frac = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        claimed_amount = frac * self._budget_remaining
        claimed_amount = max(claimed_amount, 0.0)

        row = encode_amount_only(self._base_encoded, claimed_amount, self.meta, amount_idx=AMOUNT_IDX)
        detect_proba = float(self.defender.predict_proba(row.reshape(1, -1))[0, 1])
        caught = detect_proba >= self.threshold

        self._step_count += 1
        terminated = False
        info = {"claimed_amount": claimed_amount, "detect_proba": detect_proba, "caught": caught}

        if caught:
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
