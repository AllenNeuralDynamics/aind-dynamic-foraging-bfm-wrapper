from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from data_loaders.external_bandit import (
    ExternalBanditDatasetLoader,
    _copy_dataset_with_arrays,
    apply_external_adaptation_budget,
    load_external_split_manifest,
    validate_canonical_bandit_table,
)
from utils.split_digest import bundle_split_digests


def _optional_dependency_stubs():
    modules = {}
    if importlib.util.find_spec("disentangled_rnns") is None:
        package = types.ModuleType("disentangled_rnns")
        library = types.ModuleType("disentangled_rnns.library")
        rnn_utils = types.ModuleType("disentangled_rnns.library.rnn_utils")

        class DatasetRNN:
            def __init__(
                self,
                xs,
                ys,
                *,
                y_type,
                n_classes,
                x_names,
                y_names,
                batch_size,
                batch_mode,
                rng=None,
            ):
                self._xs = np.asarray(xs)
                self._ys = np.asarray(ys)
                self.y_type = y_type
                self.n_classes = n_classes
                self.x_names = list(x_names)
                self.y_names = list(y_names)
                self.batch_size = batch_size
                self.batch_mode = batch_mode
                self.rng = rng if rng is not None else object()

            def get_all(self):
                return {"xs": self._xs, "ys": self._ys}

        rnn_utils.DatasetRNN = DatasetRNN
        library.rnn_utils = rnn_utils
        package.library = library
        modules.update(
            {
                "disentangled_rnns": package,
                "disentangled_rnns.library": library,
                "disentangled_rnns.library.rnn_utils": rnn_utils,
            }
        )
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.DictConfig = type("DictConfig", (), {})
        omegaconf.ListConfig = type("ListConfig", (), {})
        omegaconf.OmegaConf = type("OmegaConf", (), {})
        modules["omegaconf"] = omegaconf
    return mock.patch.dict(sys.modules, modules)


def _canonical_trials() -> pd.DataFrame:
    rows = []
    for subject_id in ("rat-a", "rat-b"):
        for session_id in ("s1", "s2", "s3", "s4"):
            for trial in range(4):
                rows.append(
                    {
                        "dataset_id": "demo-rats",
                        "species": "rat",
                        "subject_id": subject_id,
                        "ses_idx": session_id,
                        "trial": trial,
                        "animal_response": trial % 2,
                        "rewarded": (trial + 1) % 2,
                        "earned_reward": (trial + 1) % 2,
                    }
                )
    return pd.DataFrame(rows)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "demo-rats",
        "species": "rat",
        "subjects": [
            {
                "subject_id": subject_id,
                "adapt_session_ids": ["s1", "s3"],
                "test_session_ids": ["s2", "s4"],
            }
            for subject_id in ("rat-a", "rat-b")
        ],
    }


def _prefix_trials() -> pd.DataFrame:
    rows = []
    for subject_id in ("human-a", "human-b"):
        for trial in range(6):
            rows.append(
                {
                    "dataset_id": "demo-humans",
                    "species": "human",
                    "subject_id": subject_id,
                    "ses_idx": "main",
                    "trial": trial,
                    "animal_response": trial % 2,
                    "rewarded": (trial + 1) % 2,
                    "earned_reward": (trial + 1) % 2,
                }
            )
    return pd.DataFrame(rows)


def _prefix_manifest() -> dict:
    return {
        "schema_version": 2,
        "dataset_id": "demo-humans",
        "species": "human",
        "split_strategy": "within_session_prefix_suffix",
        "subjects": [
            {
                "subject_id": subject_id,
                "session_id": "main",
                "adapt_prefix_trials": 3,
                "total_trials": 6,
            }
            for subject_id in ("human-a", "human-b")
        ],
    }


class TestCanonicalBanditValidation(unittest.TestCase):
    def test_accepts_binary_two_arm_table(self):
        validate_canonical_bandit_table(_canonical_trials())

    def test_rejects_nonbinary_choice(self):
        df = _canonical_trials()
        df.loc[0, "animal_response"] = 2
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            validate_canonical_bandit_table(df)

    def test_rejects_duplicate_trial_key(self):
        df = pd.concat([_canonical_trials(), _canonical_trials().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_canonical_bandit_table(df)

    def test_rejects_disagreeing_reward_aliases(self):
        df = _canonical_trials()
        df.loc[0, "rewarded"] = 1 - df.loc[0, "earned_reward"]
        with self.assertRaisesRegex(ValueError, "reward aliases"):
            validate_canonical_bandit_table(df)

    def test_rejects_negative_trial_index(self):
        df = _canonical_trials()
        df.loc[0, "trial"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_canonical_bandit_table(df)


class TestExternalSplitManifest(unittest.TestCase):
    def test_rejects_unknown_schema_version(self):
        manifest = _manifest()
        manifest["schema_version"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_external_split_manifest(path)

    def test_rejects_adapt_test_overlap(self):
        manifest = _manifest()
        manifest["subjects"][0]["test_session_ids"] = ["s1"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_external_split_manifest(path)


class TestExternalBanditDatasetLoader(unittest.TestCase):
    def test_rejects_subject_ids_that_collide_after_normalization(self):
        table = _canonical_trials()
        table.loc[table["subject_id"] == "rat-a", "subject_id"] = 1
        table.loc[table["subject_id"] == "rat-b", "subject_id"] = "1"
        manifest = _manifest()
        manifest["subjects"] = [
            {
                "subject_id": 1,
                "adapt_session_ids": ["s1", "s3"],
                "test_session_ids": ["s2", "s4"],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            table.to_pickle(table_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loader = ExternalBanditDatasetLoader(
                file_path=table_path,
                split_manifest_path=manifest_path,
                batch_size=None,
                batch_mode="single",
            )
            with _optional_dependency_stubs():
                with self.assertRaisesRegex(ValueError, "collide after string"):
                    loader.load()

    def test_manifest_order_controls_subject_indices(self):
        manifest = _manifest()
        manifest["subjects"] = list(reversed(manifest["subjects"]))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    batch_size=None,
                    batch_mode="single",
                ).load()

        self.assertEqual(bundle.metadata["subject_ids"], ["rat-b", "rat-a"])
        self.assertEqual(
            bundle.metadata["subject_id_to_index"],
            {"rat-b": 0, "rat-a": 1},
        )

    def test_rejects_conflicting_species_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            loader = ExternalBanditDatasetLoader(
                file_path=table_path,
                split_manifest_path=manifest_path,
                species="mouse",
                batch_size=None,
                batch_mode="single",
            )
            with _optional_dependency_stubs():
                with self.assertRaisesRegex(ValueError, "species values must agree"):
                    loader.load()

    def test_explicit_manifest_controls_train_and_eval_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    batch_size=None,
                    batch_mode="single",
                    seed=7,
                ).load()

        self.assertEqual(bundle.metadata["dataset_id"], "demo-rats")
        self.assertEqual(bundle.metadata["species"], "rat")
        self.assertEqual(bundle.metadata["split_strategy"], "explicit_manifest")
        self.assertEqual(bundle.train_set.get_all()["xs"].shape[1], 4)
        self.assertEqual(bundle.eval_set.get_all()["xs"].shape[1], 4)
        self.assertEqual(
            bundle.metadata["adapt_session_ids_by_subject"],
            {"rat-a": ["s1", "s3"], "rat-b": ["s1", "s3"]},
        )
        self.assertEqual(
            bundle.metadata["test_session_ids_by_subject"],
            {"rat-a": ["s2", "s4"], "rat-b": ["s2", "s4"]},
        )
        self.assertEqual(
            bundle.metadata["train_session_ids"],
            ["rat-a__s1", "rat-a__s3", "rat-b__s1", "rat-b__s3"],
        )
        self.assertEqual(
            bundle.metadata["eval_session_ids"],
            ["rat-a__s2", "rat-a__s4", "rat-b__s2", "rat-b__s4"],
        )

    def test_rejects_manifest_that_does_not_cover_all_sessions(self):
        manifest = _manifest()
        manifest["subjects"][0]["test_session_ids"] = ["s2"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loader = ExternalBanditDatasetLoader(
                file_path=table_path,
                split_manifest_path=manifest_path,
                batch_size=None,
                batch_mode="single",
            )
            with _optional_dependency_stubs():
                with self.assertRaisesRegex(ValueError, "cover every retained session"):
                    loader.load()

    def test_prefix_split_warms_eval_state_without_scoring_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _prefix_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_prefix_manifest()), encoding="utf-8")

            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    batch_size=None,
                    batch_mode="single",
                ).load()

        train = bundle.train_set.get_all()
        eval_data = bundle.eval_set.get_all()
        self.assertEqual(train["xs"].shape[:2], (3, 2))
        self.assertEqual(eval_data["xs"].shape[:2], (6, 2))
        self.assertTrue(np.all(train["ys"] >= 0))
        self.assertTrue(np.all(eval_data["ys"][:3] == -1))
        self.assertTrue(np.all(eval_data["ys"][3:] >= 0))
        np.testing.assert_array_equal(
            train["xs"],
            eval_data["xs"][:3],
        )
        self.assertEqual(
            bundle.raw.groupby("subject_id")["external_split_partition"].apply(list).tolist(),
            [["adapt"] * 3 + ["test"] * 3, ["adapt"] * 3 + ["test"] * 3],
        )
        self.assertEqual(bundle.metadata["split_strategy"], "within_session_prefix_suffix")
        self.assertEqual(
            bundle.metadata["trial_partition_column"],
            "external_split_partition",
        )
        self.assertEqual(bundle.metadata["adapt_trial_partition"], "adapt")
        self.assertEqual(bundle.metadata["test_trial_partition"], "test")
        self.assertIs(bundle.train_set.rng, bundle.eval_set.rng)
        # split digests (wrapper #131) read the partition from bundle.raw: same sessions,
        # adapt prefix vs scored suffix
        digests = bundle_split_digests(bundle, heldout=True)
        context, scored = digests["heldout_context"], digests["heldout_eval"]
        self.assertEqual((context["n_sessions"], context["n_trials"]), (2, 6))
        self.assertEqual((scored["n_sessions"], scored["n_trials"]), (2, 6))
        self.assertEqual(context["sessions_sha256"], scored["sessions_sha256"])
        self.assertNotEqual(context["trials_sha256"], scored["trials_sha256"])

    def test_session_budget_uses_first_manifest_adaptation_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    adapt_sessions_per_subject=1,
                    batch_size=None,
                    batch_mode="single",
                ).load()

        self.assertEqual(bundle.train_set.get_all()["xs"].shape[1], 2)
        self.assertEqual(bundle.eval_set.get_all()["xs"].shape[1], 4)
        self.assertEqual(
            bundle.metadata["train_session_ids"],
            ["rat-a__s1", "rat-b__s1"],
        )
        selected = bundle.raw[bundle.raw["target_adaptation_selected"]]
        self.assertEqual(
            selected.groupby("subject_id")["source_ses_idx"].unique().apply(list).tolist(),
            [["s1"], ["s1"]],
        )

    def test_session_budget_rejects_incomplete_subject_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    batch_size=None,
                    batch_mode="single",
                ).load()

        bundle.metadata["session_context"] = {"per_subject": []}
        with self.assertRaisesRegex(ValueError, "map every adaptation session"):
            apply_external_adaptation_budget(bundle, adapt_sessions_per_subject=1)

    def test_dataset_copy_supports_keyword_only_array_constructor(self):
        class KeywordOnlyDataset:
            def __init__(self, *, xs, ys, **kwargs):
                self.xs = np.asarray(xs)
                self.ys = np.asarray(ys)
                vars(self).update(kwargs)

        source = KeywordOnlyDataset(
            xs=np.zeros((2, 3, 4)),
            ys=np.zeros((2, 3, 1)),
            y_type="categorical",
            n_classes=2,
            x_names=["choice", "reward"],
            y_names=["choice"],
            batch_size=None,
            batch_mode="single",
            rng=object(),
        )
        copied_xs = np.ones((2, 1, 4))
        copied_ys = np.ones((2, 1, 1))

        copied = _copy_dataset_with_arrays(source, copied_xs, copied_ys)

        np.testing.assert_array_equal(copied.xs, copied_xs)
        np.testing.assert_array_equal(copied.ys, copied_ys)
        self.assertIs(copied.rng, source.rng)

    def test_zero_session_budget_keeps_aligned_reference_tensor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    adapt_sessions_per_subject=0,
                    batch_size=None,
                    batch_mode="single",
                ).load()

        train = bundle.train_set.get_all()
        self.assertEqual(train["xs"].shape[1], 4)
        self.assertEqual(train["ys"].shape[1], 4)
        self.assertEqual(len(bundle.metadata["train_session_ids"]), 4)
        self.assertEqual(bundle.metadata["target_adaptation_session_ids"], [])
        self.assertEqual(
            bundle.metadata["adapt_sessions_per_subject_counts"],
            {"rat-a": 0, "rat-b": 0},
        )
        self.assertTrue(bundle.metadata["zero_shot_train_set_is_reference"])
        self.assertFalse(bundle.raw["target_adaptation_selected"].any())
        # the held-out context digest names the sessions adapted on (none), not the reference set
        digests = bundle_split_digests(bundle, heldout=True)
        self.assertEqual(digests["heldout_context"]["n_sessions"], 0)
        self.assertEqual(digests["heldout_eval"]["n_sessions"], 4)

if __name__ == "__main__":
    unittest.main()
