"""GLMHMMTrainer: data marshalling, GRU-identical scoring, and an end-to-end fit."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from base.types import DatasetBundle
from model_trainers.glmhmm_trainer import _pack, _sessions, score_sessions
from utils.multisubject import compute_train_eval_session_ids

try:
    import aind_dynamic_foraging_models.glm_hmm  # noqa: F401

    from model_trainers.glmhmm_trainer import GLMHMMTrainer

    HAS_GLMHMM = True
except ImportError:  # pragma: no cover - exercised only without the glmhmm extra
    HAS_GLMHMM = False


def _make_frame(subjects=("m0", "m1", "m2"), n_sessions=4, n_trials=80, seed=0, stage=True):
    """A trial frame shaped like the wrapper's raw dataframe, with ignores and blocks of
    disengagement so a two-state model has something to find."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in subjects:
        for session in range(n_sessions):
            engaged = True
            for trial in range(n_trials):
                if rng.random() < 0.05:
                    engaged = not engaged
                if engaged:
                    response = int(rng.random() < 0.6)
                else:
                    response = 2 if rng.random() < 0.7 else int(rng.integers(0, 2))
                row = {
                    "subject_id": subject,
                    "ses_idx": f"{subject}_s{session}",
                    "trial": trial,
                    "animal_response": response,
                    "earned_reward": bool(response != 2 and rng.random() < 0.4),
                }
                if stage:
                    row["current_stage_actual"] = "STAGE_FINAL" if session >= 2 else "STAGE_2"
                rows.append(row)
    return pd.DataFrame(rows)


def _split(frame):
    """The loader's per-subject even/odd session split, as bundle metadata carries it."""
    train, evals = [], []
    for _, rows in frame.groupby("subject_id", sort=False):
        ids = list(dict.fromkeys(rows["ses_idx"]))
        t, e = compute_train_eval_session_ids(ids, 2)
        train += t
        evals += e
    return {"train_session_ids": train, "eval_session_ids": evals, "ignore_policy": "include"}


class TestSessions(unittest.TestCase):
    """Session extraction keeps every trial and follows the loader's order."""

    def test_keeps_ignores_and_order(self):
        """All trials, including ignores, in the requested session order."""
        frame = _make_frame()
        records = _sessions(frame, ["m1_s2", "m0_s0"])
        self.assertEqual([r["ses_idx"] for r in records], ["m1_s2", "m0_s0"])
        self.assertEqual(len(records[0]["choices"]), 80)
        self.assertIn(2, set(np.concatenate([r["choices"] for r in records])))
        self.assertEqual(records[0]["subject_id"], "m1")
        self.assertEqual(records[0]["stage"][0], "STAGE_FINAL")

    def test_trials_are_sorted_within_a_session(self):
        """Rows out of order come back sorted by trial."""
        frame = _make_frame().sample(frac=1.0, random_state=0)
        record = _sessions(frame, ["m0_s1"])[0]
        np.testing.assert_array_equal(record["trial"], np.arange(80))

    def test_missing_columns_and_bad_codes_raise(self):
        """A frame without the required columns, or with an unknown response, fails."""
        with self.assertRaises(ValueError):
            _sessions(pd.DataFrame({"subject_id": ["m0"]}), ["s"])
        frame = _make_frame()
        frame.loc[0, "animal_response"] = 5
        with self.assertRaises(ValueError):
            _sessions(frame, ["m0_s0"])


class TestScoring(unittest.TestCase):
    """Scores go through rnn_utils and the GRU's own three-way decomposition."""

    def setUp(self):
        """Two sessions with known predictive probabilities."""
        self.records = _sessions(_make_frame(n_trials=30), ["m0_s0", "m0_s1"])
        rng = np.random.default_rng(1)
        self.log_probs = [np.log(rng.dirichlet(np.ones(3), size=len(r["choices"])))
                          for r in self.records]

    def test_matches_the_geometric_mean_of_observed_probabilities(self):
        """eval_likelihood = exp(mean log p(observed)); the log-lik sum pools exactly."""
        scores = score_sessions(self.records, self.log_probs)
        observed = np.concatenate([lp[np.arange(len(r["choices"])), r["choices"]]
                                   for r, lp in zip(self.records, self.log_probs)])
        self.assertEqual(scores["num_eval_trials"], 60)
        self.assertAlmostEqual(scores["eval_log_lik_sum"], observed.sum(), places=8)
        self.assertAlmostEqual(scores["eval_likelihood"], np.exp(observed.mean()), places=5)

    def test_decomposition_keys_and_lr_engaged(self):
        """LR_engaged renormalises over left/right on engaged trials, as for the GRU."""
        scores = score_sessions(self.records, self.log_probs)
        for key in ("eval_likelihood_LR_engaged", "eval_likelihood_engage",
                    "engage_ignore_base_rate", "engage_ignore_pr_auc"):
            self.assertIn(key, scores)
        lr = []
        for r, lp in zip(self.records, self.log_probs):
            p = np.exp(lp[:, :2])
            p /= p.sum(1, keepdims=True)
            engaged = r["choices"] != 2
            lr.append(np.log(p[np.flatnonzero(engaged), r["choices"][engaged]]))
        self.assertAlmostEqual(scores["eval_likelihood_LR_engaged"],
                               np.exp(np.concatenate(lr).mean()), places=5)

    def test_mask_restricts_the_scored_trials(self):
        """A mask scores only the selected trials, with padding still masked."""
        scores = score_sessions(self.records, self.log_probs,
                                lambda r: np.arange(len(r["choices"])) < 10)
        self.assertEqual(scores["num_eval_trials"], 20)
        labels, logits = _pack(self.records, self.log_probs)
        self.assertEqual(labels.shape, (30, 2, 1))
        self.assertEqual(logits.shape, (30, 2, 3))


@unittest.skipUnless(HAS_GLMHMM, "needs aind-dynamic-foraging-models[glmhmm]")
class TestFit(unittest.TestCase):
    """End to end on a small synthetic cohort."""

    class _Run:
        """Captures summary keys and logged objects like a W&B run."""

        def __init__(self):
            """Empty summary and log."""
            self.summary, self.logged = {}, []

        def log(self, payload, **_):
            """Record a log call."""
            self.logged.append(payload)

    def _bundle(self, ignore_policy="include"):
        """Training cohort m0-m2 and held-out cohort h0-h1, split as the loader splits."""
        train = _make_frame(("m0", "m1", "m2"), seed=0)
        heldout = _make_frame(("h0", "h1"), seed=1)
        metadata = {**_split(train), "ignore_policy": ignore_policy}
        return DatasetBundle(
            raw=train, train_set=None, eval_set=None, metadata=metadata,
            extras={"heldout_raw": heldout, "heldout_metadata": _split(heldout)},
        )

    def _trainer(self, out, **kwargs):
        """A small, fast configuration."""
        settings = dict(num_states=2, n_lags=2, num_restarts=2, max_em_iters=25,
                        output_dir=out, seed=0)
        settings.update(kwargs)
        return GLMHMMTrainer(**settings)

    def test_fit_reports_every_key_and_writes_artifacts(self):
        """Zero-shot, matched, decomposition, stage split, selection score, and files."""
        with tempfile.TemporaryDirectory() as out:
            run = self._Run()
            summary = self._trainer(out).fit(self._bundle(), loggers={"wandb": run})
            for key in (
                "heldout/eval_likelihood", "heldout/zero_shot/eval_likelihood",
                "heldout/final/eval_likelihood", "heldout/final/eval_likelihood_LR_engaged",
                "heldout/final/eval_likelihood_engage", "heldout/final/engage_ignore_pr_auc",
                "heldout/eval_log_lik_sum", "heldout/num_eval_trials",
                "heldout/final/eval_likelihood_mature", "heldout/final/eval_likelihood_nonmature",
                "heldout/train_likelihood", "glmhmm/select/eval_likelihood",
                "glmhmm/global/log_joint",
            ):
                self.assertIn(key, summary)
                self.assertIn(key, run.summary)
            for key in ("heldout/eval_likelihood", "heldout/zero_shot/eval_likelihood"):
                self.assertTrue(1 / 3 < summary[key] < 1, (key, summary[key]))
            # The GRU's zero-shot key covers every held-out session; ours covers eval only.
            self.assertNotIn("heldout/test_likelihood", summary)
            self.assertEqual(summary["heldout/zero_shot/num_eval_trials"],
                             summary["heldout/num_eval_trials"])
            # Matched scores the held-out cohort's eval (odd) sessions: 2 subjects x 2 x 80.
            self.assertEqual(summary["heldout/num_eval_trials"], 2 * 2 * 80)
            self.assertEqual(summary["heldout/num_eval_subjects"], 2)
            self.assertEqual(summary["heldout/eval_likelihood"],
                             summary["heldout/final/eval_likelihood"])
            self.assertTrue(any("heldout/per_subject" in p for p in run.logged))
            folder = Path(out) / "glmhmm"
            record = json.loads((folder / "fit.json").read_text())
            self.assertEqual(sorted(record["subjects"]), ["h0", "h1"])
            per_subject = pd.read_csv(folder / "heldout_per_subject.csv")
            self.assertEqual(int(per_subject["n_eval_trials"].sum()),
                             summary["heldout/num_eval_trials"])
            # Iteration counts are EM updates: never more than max_em_iters (25 here), and one
            # fewer than the log-joint trace, which also holds the starting point.
            self.assertLessEqual(summary["glmhmm/global/em_iters"], 25)
            self.assertEqual(summary["glmhmm/global/em_iters"],
                             len(record["global_log_joint_trace"]) - 1)
            self.assertTrue((per_subject["em_iters"] <= 25).all())
            trials = pd.read_parquet(folder / "heldout_trials.parquet")
            self.assertEqual(len(trials), 2 * 4 * 80)  # every held-out session, both splits
            probs = np.exp(trials[["logp_left", "logp_right", "logp_ignore"]].to_numpy())
            np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)
            np.testing.assert_allclose(trials[["smoothed_0", "smoothed_1"]].sum(1), 1.0,
                                       atol=1e-5)

    def test_per_subject_log_lik_sums_pool_to_the_headline(self):
        """Pooled per-subject log-lik sums and trial counts reproduce heldout/eval_likelihood."""
        with tempfile.TemporaryDirectory() as out:
            summary = self._trainer(out, num_states=1, save_posteriors=False).fit(
                self._bundle())
            per_subject = pd.read_csv(Path(out) / "glmhmm" / "heldout_per_subject.csv")
            pooled = np.exp(per_subject["eval_log_lik_sum"].sum()
                            / per_subject["n_eval_trials"].sum())
            self.assertAlmostEqual(pooled, summary["heldout/eval_likelihood"], places=5)

    def test_reference_emission_runs(self):
        """The reference parameterisation goes through the same protocol."""
        with tempfile.TemporaryDirectory() as out:
            summary = self._trainer(out, emission="reference", select_on_train_subjects=False,
                                    save_posteriors=False).fit(self._bundle())
            self.assertTrue(1 / 3 < summary["heldout/eval_likelihood"] < 1)
            self.assertNotIn("glmhmm/select/eval_likelihood", summary)

    def test_requires_ignores_included(self):
        """A two-way data config is refused: the model has an ignore outcome."""
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(ValueError):
                self._trainer(out).fit(self._bundle(ignore_policy="exclude"))

    def test_skips_heldout_without_a_cohort(self):
        """No held-out cohort: fit and selection run, held-out keys are absent."""
        bundle = self._bundle()
        bundle.extras = {}
        with tempfile.TemporaryDirectory() as out:
            summary = self._trainer(out, num_states=1).fit(bundle)
        self.assertNotIn("heldout/eval_likelihood", summary)
        self.assertIn("glmhmm/select/eval_likelihood", summary)


class TestRunnerWiring(unittest.TestCase):
    """glmhmm is a multisubject model type the runner hands the held-out cohort to."""

    def test_multisubject_type(self):
        """_is_multisubject_personalized_run accepts glmhmm."""
        from types import SimpleNamespace

        from training_runner import _is_multisubject_personalized_run

        cfg = SimpleNamespace(model=SimpleNamespace(
            type="glmhmm", architecture=SimpleNamespace(multisubject=True)))
        self.assertTrue(_is_multisubject_personalized_run(cfg))

    def test_heldout_loader_never_inherits_training_subject_ids(self):
        """Explicit training ids are cleared; an explicit held-out subset is honoured."""
        from types import SimpleNamespace

        from training_runner import _baseline_rl_heldout_loader_kwargs

        training_only = SimpleNamespace(subject_ids=["m0", "m1"], test_subject_ids=None)
        kwargs = _baseline_rl_heldout_loader_kwargs(training_only)
        self.assertIn("subject_ids", kwargs)
        self.assertIsNone(kwargs["subject_ids"])
        self.assertEqual(kwargs["split"], "heldout")
        subset = SimpleNamespace(subject_ids=["m0"], test_subject_ids=["h0", "h1"])
        self.assertEqual(_baseline_rl_heldout_loader_kwargs(subset)["subject_ids"], ["h0", "h1"])

    def test_missing_stage_column_warns_and_omits_the_split(self):
        """Without the stage column the split is absent and a warning says why."""
        if not HAS_GLMHMM:
            self.skipTest("needs aind-dynamic-foraging-models[glmhmm]")
        train = _make_frame(("m0", "m1"), stage=False)
        heldout = _make_frame(("h0",), seed=1, stage=False)
        bundle = DatasetBundle(raw=train, train_set=None, eval_set=None,
                               metadata=_split(train),
                               extras={"heldout_raw": heldout,
                                       "heldout_metadata": _split(heldout)})
        with tempfile.TemporaryDirectory() as out:
            trainer = GLMHMMTrainer(num_states=1, n_lags=2, max_em_iters=10, output_dir=out,
                                    select_on_train_subjects=False, save_posteriors=False)
            with self.assertLogs("model_trainers.glmhmm_trainer", level="WARNING") as logs:
                summary = trainer.fit(bundle)
        self.assertTrue(any("current_stage_actual" in line for line in logs.output))
        self.assertNotIn("heldout/final/eval_likelihood_mature", summary)


if __name__ == "__main__":
    unittest.main()
