"""Split digests: one identity for the data a run trained and was scored on (wrapper #131)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from utils.split_digest import (
    bundle_split_digests,
    log_split_digests,
    record_split_digests,
    split_digests,
)


def _frame(composite=False, ignored=()):
    """Two subjects, two sessions each, three trials per session.

    ``ignored`` lists (session, trial) pairs whose animal_response is 2 (no response).
    ``composite`` mimics the multisubject builder: ``ses_idx = "<subject>__<source>"`` with
    the original id kept in ``source_ses_idx``.
    """
    rows = []
    for subject, sessions in (("s1", ("a", "b")), ("s2", ("c", "d"))):
        for session in sessions:
            for trial in range(3):
                row = {
                    "subject_id": subject,
                    "ses_idx": f"{subject}__{session}" if composite else session,
                    "trial": trial,
                    "animal_response": 2 if (session, trial) in ignored else trial % 2,
                }
                if composite:
                    row["source_ses_idx"] = session
                rows.append(row)
    return pd.DataFrame(rows)


def _ids(df, sessions):
    """The ``ses_idx`` values of the given source sessions, as the bundle metadata holds them."""
    key = "source_ses_idx" if "source_ses_idx" in df.columns else "ses_idx"
    return list(dict.fromkeys(df.loc[df[key].isin(sessions), "ses_idx"]))


class SplitDigestTest(unittest.TestCase):
    def digest(self, df, sessions, policy="exclude", **kw):
        return split_digests(df, {"eval": _ids(df, sessions)}, ignore_policy=policy, **kw)["eval"]

    def test_order_of_sessions_does_not_matter(self):
        df = _frame()
        a = split_digests(df, {"eval": ["a", "c"]}, ignore_policy="exclude")["eval"]
        b = split_digests(df, {"eval": ["c", "a"]}, ignore_policy="exclude")["eval"]
        self.assertEqual(a, b)

    def test_composite_and_source_ids_give_one_digest(self):
        plain, composite = _frame(), _frame(composite=True)
        self.assertEqual(self.digest(plain, ["a", "d"]), self.digest(composite, ["a", "d"]))

    def test_different_membership_gives_different_digests(self):
        df = _frame()
        a, b = self.digest(df, ["a", "c"]), self.digest(df, ["a", "d"])
        self.assertNotEqual(a["sessions_sha256"], b["sessions_sha256"])
        self.assertNotEqual(a["trials_sha256"], b["trials_sha256"])

    def test_role_name_is_not_part_of_the_digest(self):
        # the digest names a set of data, so two runs that score it under different roles match
        df = _frame()
        d = split_digests(df, {"eval": ["a"], "heldout_eval": ["a"]}, ignore_policy="exclude")
        self.assertEqual(d["eval"], d["heldout_eval"])

    def test_ignore_policy_changes_trials_not_sessions(self):
        df = _frame(ignored={("a", 1)})
        exclude, include = self.digest(df, ["a", "c"]), self.digest(df, ["a", "c"], "include")
        self.assertEqual(exclude["sessions_sha256"], include["sessions_sha256"])
        self.assertNotEqual(exclude["trials_sha256"], include["trials_sha256"])
        self.assertEqual((exclude["n_trials"], include["n_trials"]), (5, 6))

    def test_policy_is_recorded_even_without_ignored_trials(self):
        df = _frame()
        exclude, include = self.digest(df, ["a"]), self.digest(df, ["a"], "include")
        self.assertNotEqual(exclude["trials_sha256"], include["trials_sha256"])

    def test_all_ignored_session_is_not_a_used_session_under_exclude(self):
        df = _frame(ignored={("c", 0), ("c", 1), ("c", 2)})
        exclude, include = self.digest(df, ["a", "c"]), self.digest(df, ["a", "c"], "include")
        self.assertEqual(exclude["n_sessions"], 1)
        self.assertEqual(include["n_sessions"], 2)
        self.assertNotEqual(exclude["sessions_sha256"], include["sessions_sha256"])

    def test_within_session_partition_splits_trials(self):
        df = _frame()
        df["part"] = ["adapt" if t < 2 else "test" for t in df["trial"]]
        d = split_digests(df, {"train": ["a"], "eval": ["a"]}, ignore_policy="exclude",
                          partition={"train": ("part", "adapt"), "eval": ("part", "test")})
        self.assertEqual(d["train"]["sessions_sha256"], d["eval"]["sessions_sha256"])
        self.assertNotEqual(d["train"]["trials_sha256"], d["eval"]["trials_sha256"])
        self.assertEqual((d["train"]["n_trials"], d["eval"]["n_trials"]), (2, 1))

    def test_trial_index_storage_type_does_not_matter(self):
        df = _frame()
        as_str, as_float = df.assign(trial=df["trial"].astype(str)), df.assign(trial=df["trial"].astype(float))
        self.assertEqual(self.digest(df, ["a"]), self.digest(as_str, ["a"]))
        self.assertEqual(self.digest(df, ["a"]), self.digest(as_float, ["a"]))

    def test_without_trial_column_positions_survive_filtering(self):
        # an adapt prefix and a disjoint suffix of equal length must not share a trials digest
        df = _frame().drop(columns="trial")
        df = pd.concat([df, df[df["ses_idx"] == "a"]], ignore_index=True)  # session a: 6 rows
        df = df.sort_values("ses_idx", kind="stable").reset_index(drop=True)
        df["part"] = ["test"] * len(df)
        a_rows = df.index[df["ses_idx"] == "a"]
        df.loc[a_rows[:3], "part"] = "adapt"
        d = split_digests(df, {"train": ["a"], "eval": ["a"]}, ignore_policy="include",
                          partition={"train": ("part", "adapt"), "eval": ("part", "test")})
        self.assertEqual((d["train"]["n_trials"], d["eval"]["n_trials"]), (3, 3))
        self.assertNotEqual(d["train"]["trials_sha256"], d["eval"]["trials_sha256"])

    def test_counts(self):
        d = self.digest(_frame(), ["a", "b", "c"])
        self.assertEqual((d["n_subjects"], d["n_sessions"], d["n_trials"]), (2, 3, 9))

    def test_missing_subject_id_fails_loudly(self):
        with self.assertRaises(ValueError):
            split_digests(_frame().drop(columns="subject_id"), {"eval": ["a"]}, ignore_policy="exclude")

    def test_unknown_session_id_fails_loudly(self):
        with self.assertRaises(ValueError):
            split_digests(_frame(), {"eval": ["a", "zz"]}, ignore_policy="exclude")


class BundleAndLoggingTest(unittest.TestCase):
    def test_bundle_roles_from_metadata(self):
        df = _frame(composite=True)
        bundle = SimpleNamespace(raw=df, metadata={
            "train_session_ids": _ids(df, ["a", "c"]), "eval_session_ids": _ids(df, ["b", "d"]),
            "ignore_policy": "exclude"})
        d = bundle_split_digests(bundle)
        self.assertEqual(set(d), {"train", "eval"})
        self.assertEqual(d["train"], split_digests(df, {"train": _ids(df, ["a", "c"])},
                                                   ignore_policy="exclude")["train"])
        held = bundle_split_digests(bundle, heldout=True)
        self.assertEqual(set(held), {"heldout_context", "heldout_eval"})

    def test_absent_role_is_not_digested_but_empty_is(self):
        df = _frame()
        only_eval = SimpleNamespace(raw=df, metadata={"eval_session_ids": ["b"]})
        self.assertEqual(set(bundle_split_digests(only_eval)), {"eval"})
        empty = SimpleNamespace(raw=df, metadata={"train_session_ids": [], "eval_session_ids": ["b"]})
        self.assertEqual(bundle_split_digests(empty)["train"]["n_sessions"], 0)
        with self.assertRaises(ValueError):
            bundle_split_digests(SimpleNamespace(raw=df, metadata={}))

    def test_adaptation_budget_names_the_adapted_sessions(self):
        # at K = 0 train_session_ids keeps every session as a reference set; none is adapted on
        df = _frame()
        bundle = SimpleNamespace(raw=df, metadata={
            "train_session_ids": ["a", "c"], "eval_session_ids": ["b", "d"],
            "target_adaptation_session_ids": []})
        d = bundle_split_digests(bundle, heldout=True)
        self.assertEqual(d["heldout_context"]["n_sessions"], 0)
        self.assertEqual(d["heldout_eval"]["n_sessions"], 2)

    def test_zero_shot_finetune_has_no_context(self):
        # snapshot fine-tune at K = 0 keeps train_session_ids as a reference set (no target_* key)
        df = _frame()
        bundle = SimpleNamespace(raw=df, metadata={
            "train_session_ids": ["a", "c"], "eval_session_ids": ["b", "d"],
            "adapt_sessions_per_subject": 0})
        d = bundle_split_digests(bundle, heldout=True)
        self.assertEqual(d["heldout_context"]["n_sessions"], 0)
        self.assertEqual(bundle_split_digests(bundle)["train"]["n_sessions"], 2)  # training roles untouched

    def test_bundle_within_session_partition(self):
        df = _frame()
        df["external_split_partition"] = ["adapt" if t < 2 else "test" for t in df["trial"]]
        bundle = SimpleNamespace(raw=df, metadata={
            "train_session_ids": ["a"], "eval_session_ids": ["a"],
            "trial_partition_column": "external_split_partition",
            "adapt_trial_partition": "adapt", "test_trial_partition": "test"})
        d = bundle_split_digests(bundle)
        self.assertEqual((d["train"]["n_trials"], d["eval"]["n_trials"]), (2, 1))

    def test_log_writes_summary_keys(self):
        run = SimpleNamespace(summary={})
        d = split_digests(_frame(), {"train": ["a"]}, ignore_policy="exclude")
        log_split_digests(run, d)
        self.assertEqual(run.summary["split/train/sessions_sha256"], d["train"]["sessions_sha256"])
        self.assertEqual(run.summary["split/train/n_trials"], 3)
        log_split_digests(None, d)  # no run: nothing to do

    def test_record_never_raises(self):
        run = SimpleNamespace(summary={})
        bad = lambda: split_digests(_frame(), {"eval": ["zz"]}, ignore_policy="exclude")  # noqa: E731
        with self.assertLogs("utils.split_digest", level="WARNING"):
            self.assertIsNone(record_split_digests(run, bad))
        self.assertIn("ValueError", run.summary["split/error"])
        good = record_split_digests(run, lambda: split_digests(_frame(), {"eval": ["a"]},
                                                               ignore_policy="exclude"))
        self.assertEqual(run.summary["split/eval/n_sessions"], good["eval"]["n_sessions"])


class RunnerWiringTest(unittest.TestCase):
    """run_training logs the bundle's split unless the trainer logs what it used itself."""

    def _run(self, trainer, model_type="fake", data=None, policy="exclude"):
        import training_runner
        from omegaconf import OmegaConf

        def _bundle(train, scored):
            df = _frame(composite=True)
            meta = {"train_session_ids": _ids(df, train), "eval_session_ids": _ids(df, scored)}
            if policy is not None:
                meta["ignore_policy"] = policy
            return SimpleNamespace(raw=df, train_set=None, eval_set=None, extras={}, metadata=meta)

        bundle, heldout = _bundle(["a", "c"], ["b", "d"]), _bundle(["a"], ["b", "c", "d"])
        run = SimpleNamespace(name="run", summary={}, config={})
        config = OmegaConf.create({"data": data or {"type": "fake"}, "model": {"type": model_type},
                                   "seed": 0})

        def _instantiate(cfg, *args, **kwargs):
            if cfg is config.data:  # the held-out loader is the data config with split="heldout"
                chosen = heldout if kwargs.get("split") == "heldout" else bundle
                return SimpleNamespace(load=lambda: chosen)
            return trainer

        with mock.patch.object(training_runner, "instantiate", side_effect=_instantiate):
            training_runner.run_training(config, wandb_run=run)
        return run, bundle

    def test_logs_bundle_split(self):
        trainer = SimpleNamespace(fit=lambda bundle, loggers=None: {})
        run, bundle = self._run(trainer)
        for role in ("train", "eval"):
            self.assertEqual(run.summary[f"split/{role}/n_sessions"], 2)
        self.assertEqual(bundle.metadata["split_digests"]["train"]["sessions_sha256"],
                         run.summary["split/train/sessions_sha256"])

    def test_glmhmm_heldout_bundle_is_digested(self):
        trainer = SimpleNamespace(fit=lambda bundle, loggers=None: {})
        data = {"type": "mice_snapshot", "mature_only": False, "min_sessions": 1, "heldout_every_n": 5}
        run, _ = self._run(trainer, model_type="glmhmm", data=data)
        self.assertNotIn("split/error", run.summary)
        self.assertEqual(run.summary["split/heldout_context/n_sessions"], 1)
        self.assertEqual(run.summary["split/heldout_eval/n_sessions"], 3)
        self.assertEqual(run.summary["split/train/n_sessions"], 2)

    def test_glmhmm_missing_policy_means_include(self):
        trainer = SimpleNamespace(fit=lambda bundle, loggers=None: {})
        include, _ = self._run(trainer, model_type="glmhmm", policy=None)
        exclude, _ = self._run(trainer, model_type="fake", policy=None)
        explicit, _ = self._run(trainer, model_type="fake", policy="include")
        self.assertEqual(include.summary["split/train/trials_sha256"], explicit.summary["split/train/trials_sha256"])
        self.assertNotEqual(include.summary["split/train/trials_sha256"], exclude.summary["split/train/trials_sha256"])

    def test_trainer_that_logs_its_own_split_is_left_alone(self):
        trainer = SimpleNamespace(fit=lambda bundle, loggers=None: {}, logs_own_split_digests=True)
        run, bundle = self._run(trainer)
        self.assertFalse([k for k in run.summary if k.startswith("split/")])
        self.assertNotIn("split_digests", bundle.metadata)


class EvaluatorLoadTest(unittest.TestCase):
    """Held-out data an evaluator loads itself (after the runner's digest point) is digested."""

    def test_evaluators_digest_the_data_they_load(self):
        import tempfile
        from pathlib import Path

        from omegaconf import OmegaConf

        import evaluation.disrnn_evaluation as de
        import evaluation.gru_evaluation as ge

        df = _frame()
        loaded = {"df_test": df, "test_subject_ids": ["s1", "s2"], "dataset_test": None,
                  "xs_test": None, "ys_test": None}
        for module, loader, evaluate in (
            (ge, "load_gru_heldout_subject_data", ge.evaluate_gru_on_heldout_subjects),
            (de, "load_disrnn_heldout_subject_data", de.evaluate_disrnn_on_heldout_subjects),
        ):
            with self.subTest(evaluator=evaluate.__name__), tempfile.TemporaryDirectory() as out:
                (Path(out) / "params.json").write_text("{}")
                cfg = OmegaConf.create({
                    "data": {"heldout_eval": True, "test_subject_ids": ["s1", "s2"], "ignore_policy": "exclude"},
                    "model": {"architecture": {"multisubject": False}, "output_dir": out}})
                run = SimpleNamespace(summary={}, log=lambda *a, **k: None)
                with mock.patch.object(module, loader, return_value=loaded):
                    try:  # placeholder params: evaluation fails after the load, which is enough here
                        # the runner turns metric logging off; provenance is still recorded
                        evaluate(cfg, wandb_run=run, log_to_wandb=False)
                    except Exception:  # noqa: BLE001
                        pass
                self.assertEqual(run.summary["split/heldout_eval/n_sessions"], 4)
                self.assertNotIn("split/error", run.summary)


if __name__ == "__main__":
    unittest.main()
