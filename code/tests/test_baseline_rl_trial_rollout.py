"""Unit tests for per-trial baseline-RL rollouts."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from post_training_analysis.baseline_rl_trial_rollout import pooled_likelihood, rollout_baseline_rl

FIT = {
    "agent_class": "ForagerQLearning",
    "agent_kwargs": {"number_of_learning_rate": 1, "number_of_forget_rate": 1,
                     "choice_kernel": "none", "action_selection": "softmax"},
    "fitted_params": {"learn_rate": 0.3, "forget_rate_unchosen": 0.1, "biasL": 0.0,
                      "softmax_inverse_temperature": 5.0},
}


def _trials() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for subject, ses, n in (("m1", "m1__a", 30), ("m1", "m1__b", 25), ("m2", "m2__a", 40)):
        for k in range(1, n + 1):
            rows.append({"subject_id": subject, "ses_idx": ses, "trial_idx": k,
                         "action": int(rng.integers(0, 2)), "reward": float(rng.integers(0, 2))})
    return pd.DataFrame(rows)


class TestBaselineRlTrialRollout(unittest.TestCase):
    def setUp(self) -> None:
        self.trials = _trials()
        self.output = {"fitted_params_per_subject": {"m1": FIT, "m2": FIT}}

    def test_one_row_per_trial_in_input_order(self) -> None:
        out = rollout_baseline_rl(self.output, self.trials)
        self.assertEqual(len(out), len(self.trials))
        pd.testing.assert_frame_equal(out[["ses_idx", "trial_idx"]], self.trials[["ses_idx", "trial_idx"]])
        np.testing.assert_allclose(out["p_rl_left"] + out["p_rl_right"], 1.0)

    def test_q_values_start_at_initial_value_and_respond_to_reward(self) -> None:
        out = rollout_baseline_rl(self.output, self.trials)
        first = out.groupby("ses_idx").head(1)
        np.testing.assert_allclose(first[["q_rl_left", "q_rl_right"]].to_numpy(), first[["q_rl_left"]].to_numpy().repeat(2, 1))
        self.assertGreater(out["q_rl_right"].std(), 0)

    def test_two_learning_rate_agent_gets_q_values_that_reproduce_its_policy(self) -> None:
        fit = {"agent_class": "ForagerQLearning",
               "agent_kwargs": {"number_of_learning_rate": 2, "number_of_forget_rate": 1,
                                "choice_kernel": "none", "action_selection": "softmax"},
               "fitted_params": {"learn_rate_rew": 0.4, "learn_rate_unrew": 0.05, "forget_rate_unchosen": 0.2,
                                 "biasL": 0.1, "softmax_inverse_temperature": 4.0}}
        out = rollout_baseline_rl({"fitted_params_per_subject": {"m1": fit, "m2": fit}}, self.trials)
        self.assertIn("q_rl_right", out.columns)
        params = fit["fitted_params"]
        logit = params["softmax_inverse_temperature"] * (out["q_rl_right"] - out["q_rl_left"]) - params["biasL"]
        np.testing.assert_allclose(1 / (1 + np.exp(-logit)), out["p_rl_right"], atol=1e-12)

    def test_parallel_matches_serial(self) -> None:
        serial = rollout_baseline_rl(self.output, self.trials)
        parallel = rollout_baseline_rl(self.output, self.trials, workers=2)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_missing_subject_fit_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            rollout_baseline_rl({"fitted_params_per_subject": {"m1": FIT}}, self.trials)

    def test_pooled_likelihood(self) -> None:
        self.assertAlmostEqual(pooled_likelihood(np.array([1, 0]), np.array([0.8, 0.4])), float(np.sqrt(0.8 * 0.6)))


if __name__ == "__main__":
    unittest.main()
