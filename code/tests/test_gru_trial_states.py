"""Unit tests for the per-trial GRU state table."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from post_training_analysis.gru_trial_states import (
    build_trial_states_table,
    pooled_trial_likelihood,
    readout_reconstruction_error,
    state_columns,
)

HIDDEN = 3
READOUT_W = np.array([[1.0, -1.0], [0.5, 0.0], [0.0, 2.0]])
READOUT_B = np.array([0.1, -0.2])


def _output_df() -> pd.DataFrame:
    """Two sessions, rows deliberately out of order, one ignored trial."""
    rng = np.random.default_rng(0)
    rows = []
    for ses, subject, curriculum, n in (("s1", "m1", "Coupled Baiting", 5), ("s2", "m2", "Uncoupled Baiting", 4)):
        for trial in range(1, n + 1):
            state = rng.normal(size=HIDDEN)
            logits = state @ READOUT_W + READOUT_B
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            rows.append({
                "ses_idx": ses, "subject_id": subject, "trial": trial,
                "animal_response": 2 if (ses, trial) == ("s1", 3) else int(rng.integers(0, 2)),
                "earned_reward": bool(rng.integers(0, 2)),
                "curriculum_name": curriculum,
                "reward_probabilityL": 0.1 * trial,
                **{f"latent_{i}": state[i] for i in range(HIDDEN)},
                "choice_logit_0": logits[0], "choice_logit_1": logits[1],
                "choice_prob_0": probs[0], "choice_prob_1": probs[1],
            })
    df = pd.DataFrame(rows)
    return df.iloc[::-1].reset_index(drop=True)


class TestGruTrialStates(unittest.TestCase):
    def setUp(self) -> None:
        self.output_df = _output_df()
        self.table = build_trial_states_table(
            self.output_df, metadata={}, extra_columns=("reward_probabilityL",)
        )

    def test_drops_ignored_trials_and_keeps_every_choice_trial(self) -> None:
        self.assertEqual(len(self.table), 8)
        self.assertEqual(self.table.groupby("ses_idx")["trial_idx"].max().to_dict(), {"s1": 4, "s2": 4})

    def test_states_stay_attached_to_their_own_trial(self) -> None:
        source = self.output_df.set_index(["ses_idx", "trial"])
        for row in self.table.itertuples():
            expected = source.loc[(row.ses_idx, row.trial)]
            self.assertAlmostEqual(row.rnn_state_0, expected["latent_0"], places=5)
            self.assertEqual(row.action, expected["animal_response"])
            self.assertAlmostEqual(row.reward_probabilityL, expected["reward_probabilityL"])

    def test_columns_follow_the_state_space_schema(self) -> None:
        for column in ("subject_id", "ses_idx", "trial_idx", "action", "reward",
                       "p_model1_left", "p_model1_right", "prev_action", "prev_outcome",
                       "session_curriculum_name", "logit_diff"):
            self.assertIn(column, self.table.columns)
        self.assertEqual(state_columns(self.table), ["rnn_state_0", "rnn_state_1", "rnn_state_2"])
        self.assertEqual(set(self.table["session_curriculum_name"]), {"Coupled Baiting", "Uncoupled Baiting"})

    def test_pooled_likelihood_matches_manual_formula(self) -> None:
        chosen = np.where(self.table["action"] == 1, self.table["p_model1_right"], self.table["p_model1_left"])
        self.assertAlmostEqual(pooled_trial_likelihood(self.table), float(np.exp(np.mean(np.log(chosen)))))

    def test_readout_reconstructs_logits_from_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "net/~/readout": {"w": READOUT_W.tolist(), "b": READOUT_B.tolist()},
            }))
            self.assertLess(readout_reconstruction_error(self.table, params_path), 1e-5)


if __name__ == "__main__":
    unittest.main()
