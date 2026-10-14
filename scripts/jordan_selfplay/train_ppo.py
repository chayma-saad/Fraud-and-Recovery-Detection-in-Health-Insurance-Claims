"""Trains the PPO fraudster agent against the static defender_v1 --
Jordan-dataset equivalent of scripts/selfplay/train_ppo.py."""
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from common import (
    PROJECT_ROOT, load_encoder_meta, DEFENDER_V1_PATH,
    SYNTHETIC_CLAIMS_PATH, PPO_MODEL_PATH, TRAINING_LOG_PATH,
    ADVERSARIAL_EXAMPLES_PATH, load_raw_features, attach_sampled_extras,
)
from fraud_env import ClaimFraudEnv, MAX_STEPS

TOTAL_TIMESTEPS = 200_000
EVAL_EVERY = 10_000
EVAL_EPISODES = 300


def load_defender():
    defender = XGBClassifier()
    defender.load_model(str(DEFENDER_V1_PATH))
    with open(PROJECT_ROOT / "models/jordan_selfplay/defender_v1_meta.json") as f:
        meta = json.load(f)
    return defender, meta["threshold"]


def evaluate_policy_attack_success(env, model, n_episodes, rng_seed=0, collect_examples=False):
    successes = 0
    kept_fractions = []
    examples = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=rng_seed + ep)
        terminated = truncated = False
        caught_this_ep = False
        ep_rows = []
        while not (terminated or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info["caught"]:
                caught_this_ep = True
            elif collect_examples:
                ep_rows.append(info["claimed_amount"])
        if not caught_this_ep:
            successes += 1
            if collect_examples:
                examples.append({
                    "claim_idx": env._claim_idx, "n_sub_claims": len(ep_rows),
                    "amounts": ep_rows, "kept_fraction": info["kept_fraction"],
                })
        kept_fractions.append(info["kept_fraction"])
    return successes / n_episodes, float(np.mean(kept_fractions)), examples


class AttackSuccessLoggerCallback(BaseCallback):
    def __init__(self, eval_env, eval_every, eval_episodes, log_path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.log_path = log_path
        self.rows = []
        self._last_eval = 0

    def _on_step(self):
        if self.num_timesteps - self._last_eval >= self.eval_every:
            self._last_eval = self.num_timesteps
            success_rate, kept_frac, _ = evaluate_policy_attack_success(
                self.eval_env, self.model, self.eval_episodes, rng_seed=10_000 + self.num_timesteps,
            )
            self.rows.append({"timesteps": self.num_timesteps, "attack_success_rate": success_rate,
                               "mean_kept_fraction": kept_frac})
            print(f"[{self.num_timesteps:>7,}] attack success rate: {success_rate:.1%}, "
                  f"mean budget kept: {kept_frac:.1%}")
            pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
        return True


def main():
    synthetic = pd.read_csv(SYNTHETIC_CLAIMS_PATH)
    meta = load_encoder_meta()
    defender, threshold = load_defender()

    real_df = load_raw_features()
    real_train = real_df[real_df["claim_year"] <= 2015]
    rng = np.random.default_rng(7)
    synthetic = attach_sampled_extras(synthetic, real_train, rng)

    print(f"Loaded defender (threshold={threshold:.4f}) and {len(synthetic):,} synthetic claims")

    env = ClaimFraudEnv(synthetic, meta, defender, threshold, seed=1)
    eval_env = ClaimFraudEnv(synthetic, meta, defender, threshold, seed=2)

    print("\n--- Baseline: random policy ---")
    random_success, random_kept, _ = evaluate_policy_attack_success(eval_env, None, EVAL_EPISODES, rng_seed=0)
    print(f"Random policy attack success rate: {random_success:.1%}, mean budget kept: {random_kept:.1%}")

    model = PPO("MlpPolicy", env, verbose=0, seed=42, n_steps=1024, batch_size=256, learning_rate=3e-4)
    callback = AttackSuccessLoggerCallback(eval_env, EVAL_EVERY, EVAL_EPISODES, TRAINING_LOG_PATH)

    print(f"\n--- Training PPO for {TOTAL_TIMESTEPS:,} timesteps ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    PPO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(PPO_MODEL_PATH))
    print(f"Saved: {PPO_MODEL_PATH}")

    print("\n--- Final trained policy vs. random baseline ---")
    final_success, final_kept, examples = evaluate_policy_attack_success(
        eval_env, model, EVAL_EPISODES, rng_seed=99_999, collect_examples=True,
    )
    print(f"Random policy:  attack success rate {random_success:.1%}, mean kept {random_kept:.1%}")
    print(f"Trained policy: attack success rate {final_success:.1%}, mean kept {final_kept:.1%}")

    ex_rows = []
    for ex in examples:
        for amt in ex["amounts"]:
            ex_rows.append({"claim_idx": ex["claim_idx"], "n_sub_claims": ex["n_sub_claims"],
                             "sub_claim_amount": amt, "kept_fraction": ex["kept_fraction"]})
    pd.DataFrame(ex_rows).to_csv(ADVERSARIAL_EXAMPLES_PATH, index=False)
    print(f"Saved {len(ex_rows):,} successful evasive sub-claims: {ADVERSARIAL_EXAMPLES_PATH}")

    with open(PROJECT_ROOT / "models/jordan_selfplay/ppo_summary.json", "w") as f:
        json.dump({
            "random_policy_attack_success_rate": random_success,
            "random_policy_mean_kept_fraction": random_kept,
            "trained_policy_attack_success_rate": final_success,
            "trained_policy_mean_kept_fraction": final_kept,
            "total_timesteps": TOTAL_TIMESTEPS,
        }, f, indent=2)


if __name__ == "__main__":
    main()
