"""Unit tests for the per-trial GRU state table."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from post_training_analysis.gru_trial_states import (
    RESPONSE_COLUMNS,
    build_trial_states_table,
    pooled_trial_likelihood,
    readout_reconstruction_error,
    state_columns,
)

HIDDEN = 3
READOUT_W = np.array([[1.0, -1.0], [0.5, 0.0], [0.0, 2.0]])
READOUT_B = np.array([0.1, -0.2])
# A 3-action head: column 2 is the ignore logit.
READOUT_W3 = np.array([[1.0, -1.0, 0.3], [0.5, 0.0, -0.7], [0.0, 2.0, 0.2]])
READOUT_B3 = np.array([0.1, -0.2, -1.0])


def _output_df(
    readout_w: np.ndarray = READOUT_W,
    readout_b: np.ndarray = READOUT_B,
    s1_trials: int = 5,
    ignored_trial: int = 3,
) -> pd.DataFrame:
    """Two sessions, rows deliberately out of order, one ignored trial."""
    rng = np.random.default_rng(0)
    rows = []
    for ses, subject, curriculum, n in (("s1", "m1", "Coupled Baiting", s1_trials), ("s2", "m2", "Uncoupled Baiting", 4)):
        for trial in range(1, n + 1):
            state = rng.normal(size=HIDDEN)
            logits = state @ readout_w + readout_b
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            rows.append({
                "ses_idx": ses, "subject_id": subject, "trial": trial,
                "animal_response": 2 if (ses, trial) == ("s1", ignored_trial) else int(rng.integers(0, 2)),
                "earned_reward": bool(rng.integers(0, 2)),
                "curriculum_name": curriculum,
                "reward_probabilityL": 0.1 * trial,
                **{f"latent_{i}": state[i] for i in range(HIDDEN)},
                **{f"choice_logit_{k}": logits[k] for k in range(len(logits))},
                **{f"choice_prob_{k}": probs[k] for k in range(len(probs))},
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
        self.assertNotIn("engaged_idx", self.table.columns)
        self.assertNotIn("p_model1_ignore", self.table.columns)
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

    def test_response_columns_are_absent_without_timing_features(self) -> None:
        for column in RESPONSE_COLUMNS:
            self.assertNotIn(column, self.table.columns)

    def test_response_columns_stay_attached_to_their_own_trial(self) -> None:
        output_df = self.output_df.assign(
            reaction_time=0.1 * self.output_df["trial"],
            log_reaction_time=-self.output_df["trial"].astype(float),
        )
        table = build_trial_states_table(output_df, metadata={})
        source = output_df.set_index(["ses_idx", "trial"])
        for row in table.itertuples():
            expected = source.loc[(row.ses_idx, row.trial)]
            self.assertAlmostEqual(row.reaction_time, expected["reaction_time"])
            self.assertAlmostEqual(row.log_reaction_time, expected["log_reaction_time"])
        self.assertNotIn("n_lick_left", table.columns)


class TestGruTrialStatesThreeAction(unittest.TestCase):
    """``ignore_policy=include``: the GRU steps through the ignored trial, so the table keeps it."""

    def setUp(self) -> None:
        self.output_df = _output_df(READOUT_W3, READOUT_B3)
        self.table = build_trial_states_table(
            self.output_df, metadata={}, extra_columns=("reward_probabilityL",)
        )

    def test_keeps_every_timestep_with_the_ignored_trial_in_place(self) -> None:
        self.assertEqual(len(self.table), 9)
        s1 = self.table[self.table["ses_idx"] == "s1"].sort_values("trial_idx")
        self.assertEqual(s1["trial_idx"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(s1["trial"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(s1["action"].tolist()[2], 2)
        self.assertEqual(s1["engaged_idx"].tolist()[:2] + s1["engaged_idx"].tolist()[3:], [1, 2, 3, 4])
        self.assertTrue(np.isnan(s1["engaged_idx"].tolist()[2]))

    def test_ignored_row_carries_session_fields_and_ignore_probability(self) -> None:
        row = self.table[self.table["action"] == 2].iloc[0]
        self.assertEqual((row.subject_id, row.session_id, row.curriculum_name), ("m1", "s1", "Coupled Baiting"))
        self.assertEqual(row.session_idx, 1)
        self.assertTrue(np.isnan(row.prev_action))
        probs = self.table[["p_model1_left", "p_model1_right", "p_model1_ignore"]].sum(axis=1)
        np.testing.assert_allclose(probs, 1.0, rtol=1e-6)

    def test_session_with_only_ignored_trials_keeps_its_ids(self) -> None:
        output_df = _output_df(READOUT_W3, READOUT_B3)
        output_df.loc[output_df["ses_idx"] == "s2", "animal_response"] = 2
        table = build_trial_states_table(output_df, metadata={})
        s2 = table[table["ses_idx"] == "s2"]
        self.assertEqual(len(s2), 4)
        self.assertEqual(set(s2["session_id"]), {"s2"})
        self.assertEqual(set(s2["subject_id"]), {"m2"})

    def test_engaged_history_skips_the_ignored_trial(self) -> None:
        # Long enough to clear add_candidate_variables' 10-trial history warm-up.
        table = build_trial_states_table(
            _output_df(READOUT_W3, READOUT_B3, s1_trials=15, ignored_trial=13), metadata={}
        )
        s1 = table[table["ses_idx"] == "s1"].set_index("trial")
        self.assertEqual(s1.loc[13, "action"], 2)
        self.assertEqual(s1.loc[14, "prev_action"], s1.loc[12, "action"])
        self.assertEqual(s1.loc[14, "prev_outcome"], s1.loc[12, "reward"])

    def test_states_stay_attached_to_their_own_trial(self) -> None:
        source = self.output_df.set_index(["ses_idx", "trial"])
        for row in self.table.itertuples():
            expected = source.loc[(row.ses_idx, row.trial)]
            self.assertAlmostEqual(row.rnn_state_0, expected["latent_0"], places=5)
            self.assertAlmostEqual(row.p_model1_ignore, expected["choice_prob_2"])
            self.assertEqual(row.action, expected["animal_response"])

    def test_pooled_likelihood_scores_all_three_classes(self) -> None:
        probs = self.table[["p_model1_left", "p_model1_right", "p_model1_ignore"]].to_numpy()
        chosen = probs[np.arange(len(self.table)), self.table["action"].to_numpy()]
        self.assertAlmostEqual(pooled_trial_likelihood(self.table), float(np.exp(np.mean(np.log(chosen)))))

    def test_readout_reconstructs_all_three_logits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "net/~/readout": {"w": READOUT_W3.tolist(), "b": READOUT_B3.tolist()},
            }))
            self.assertLess(readout_reconstruction_error(self.table, params_path), 1e-5)


if __name__ == "__main__":
    unittest.main()
