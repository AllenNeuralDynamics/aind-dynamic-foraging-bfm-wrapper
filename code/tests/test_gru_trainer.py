"""Smoke tests for GruTrainer."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import data_loaders.disrnn_dataset as dl
    import numpy as np
    import pandas as pd
    from disentangled_rnns.library import rnn_utils
    from omegaconf import OmegaConf

    from base.types import DatasetBundle
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.gru_trainer import (
        GruTrainer,
        _validate_multisubject_dataset_inputs,
    )
    from model_trainers.base_multisubject_trainer import BaseMultisubjectTrainer
    from utils.gru_evaluation import evaluate_gru_on_heldout_subjects
    from utils.multisubject import (
        build_subject_index_maps,
        compute_train_eval_session_ids,
        merge_datasets_with_subject_index,
    )

    GRU_DEPS_AVAILABLE = True
    GRU_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    GRU_DEPS_AVAILABLE = False
    GRU_IMPORT_ERROR = exc


@unittest.skipUnless(
    GRU_DEPS_AVAILABLE,
    f"GRU trainer dependencies unavailable: {GRU_IMPORT_ERROR}",
)
class TestGruTrainer(unittest.TestCase):
    """Test suite for GruTrainer."""

    def setUp(self):
        self.loader = SyntheticCognitiveAgents(
            task={
                "type": "random_walk",
                "reward_baiting": False,
                "p_min": 0.0,
                "p_max": 1.0,
                "sigma": 0.15,
                "mean": 0,
                "num_trials": 100,
                "seed": 42,
            },
            agent={
                "agent_class": "ForagerQLearning",
                "agent_kwargs": {
                    "number_of_learning_rate": 2,
                    "number_of_forget_rate": 1,
                    "choice_kernel": "none",
                    "action_selection": "softmax",
                },
                "agent_params": {
                    "learn_rate_rew": 0.5,
                    "learn_rate_unrew": 0.1,
                    "forget_rate_unchosen": 0.05,
                    "softmax_inverse_temperature": 5.0,
                    "biasL": 0.0,
                },
                "agent_params_session_var": {
                    "biasL": {
                        "type": "gaussian",
                        "mean": 0,
                        "std": 1,
                    }
                },
                "seed": 42,
            },
            num_trials=100,
            num_sessions=6,
            eval_every_n=2,
        )
        self.bundle = self.loader.load()
        self.multisubject_bundle = self._make_multisubject_bundle()
        self.output_dir = Path(tempfile.mkdtemp(prefix="gru_trainer_test_"))

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _make_multisubject_bundle(self) -> DatasetBundle:
        raw_df = self.bundle.raw.copy()
        raw_df["subject_id"] = np.where(raw_df["ses_idx"].astype(int) < 3, 711041, 793446)
        raw_df["source_ses_idx"] = raw_df["ses_idx"]
        raw_df["ses_idx"] = raw_df.apply(
            lambda row: f"{int(row['subject_id'])}__{int(row['source_ses_idx'])}",
            axis=1,
        )

        ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
            [711041, 793446]
        )

        full_datasets = []
        train_datasets = []
        eval_datasets = []
        ordered_frames = []
        subject_indices = []
        full_session_ids = []
        train_session_ids = []
        eval_session_ids = []
        session_max_index_by_subject_index = []
        session_context_rows = []

        for subject_id in ordered_subject_ids:
            subject_df = raw_df[raw_df["subject_id"] == subject_id].copy()
            subject_df = subject_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
            session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
            source_session_ids = list(dict.fromkeys(subject_df["source_ses_idx"].tolist()))
            session_index_by_session_id = {
                str(session_id): int(index)
                for index, session_id in enumerate(session_ids, start=1)
            }
            train_ids, eval_ids = compute_train_eval_session_ids(session_ids, eval_every_n=2)
            subject_df["subject_index"] = subject_id_to_index[subject_id]
            subject_df["subject_session_index"] = subject_df["ses_idx"].map(
                lambda session_id: session_index_by_session_id[str(session_id)]
            )
            subject_df["subject_max_session_index"] = int(len(session_ids))

            dataset = dl.create_disrnn_dataset(
                subject_df,
                ignore_policy="exclude",
                batch_size=None,
                batch_mode="single",
            )
            dataset_train, dataset_eval = rnn_utils.split_dataset(dataset, eval_every_n=2)

            full_datasets.append(dataset)
            train_datasets.append(dataset_train)
            eval_datasets.append(dataset_eval)
            ordered_frames.append(subject_df)
            subject_indices.append(subject_id_to_index[subject_id])
            full_session_ids.extend(session_ids)
            train_session_ids.extend(train_ids)
            eval_session_ids.extend(eval_ids)
            session_max_index_by_subject_index.append(int(len(session_ids)))
            session_context_rows.append(
                {
                    "subject_id": subject_id,
                    "subject_index": int(subject_id_to_index[subject_id]),
                    "ordered_session_ids": [str(session_id) for session_id in session_ids],
                    "ordered_source_session_ids": [
                        str(session_id) for session_id in source_session_ids
                    ],
                }
            )

        merged_dataset = merge_datasets_with_subject_index(full_datasets, subject_indices)
        merged_train = merge_datasets_with_subject_index(train_datasets, subject_indices)
        merged_eval = merge_datasets_with_subject_index(eval_datasets, subject_indices)
        merged_raw_df = pd.concat(ordered_frames, ignore_index=True)
        merged_raw_df["ses_idx"] = pd.Categorical(
            merged_raw_df["ses_idx"],
            categories=full_session_ids,
            ordered=True,
        )
        merged_raw_df = merged_raw_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
        merged_raw_df["ses_idx"] = merged_raw_df["ses_idx"].astype(str)

        return DatasetBundle(
            raw=merged_raw_df,
            train_set=merged_train,
            eval_set=merged_eval,
            metadata={
                "ignore_policy": "exclude",
                "eval_every_n": 2,
                "multisubject": True,
                "subject_ids": ordered_subject_ids,
                "subject_id_to_index": subject_id_to_index,
                "index_to_subject_id": index_to_subject_id,
                "num_subjects": len(ordered_subject_ids),
                "session_max_index_by_subject_index": session_max_index_by_subject_index,
                "session_context": {
                    "indexing": "1_based",
                    "per_subject": session_context_rows,
                },
                "num_trials": len(merged_raw_df),
                "num_sessions": len(full_session_ids),
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
            },
            extras={"dataset": merged_dataset},
        )

    def test_instantiation(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            seed=42,
            output_dir=str(self.output_dir),
        )
        self.assertEqual(trainer.architecture["hidden_size"], 8)
        self.assertEqual(trainer.architecture["num_layers"], 1)
        self.assertEqual(trainer.seed, 42)

    def test_fit_basic(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 5,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "save_output_df": True,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.bundle)

        self.assertIn("initial_evaluations", output)
        self.assertIn("before_training", output["initial_evaluations"])
        self.assertIn("likelihood", output)
        self.assertIn("likelihood_train", output)
        self.assertIn("training_time", output)
        self.assertIn("split_examples", output)
        self.assertIn("random_key", output)
        before_training = output["initial_evaluations"]["before_training"]
        self.assertTrue(Path(before_training["params_path"]).exists())
        self.assertNotIn("output_df_path", before_training)
        self.assertIn("split_examples", before_training)
        self.assertIsInstance(output["likelihood"], float)
        self.assertIsInstance(output["likelihood_train"], float)
        self.assertGreaterEqual(output["likelihood"], 0.0)
        self.assertLessEqual(output["likelihood"], 1.0)
        self.assertGreaterEqual(output["likelihood_train"], 0.0)
        self.assertLessEqual(output["likelihood_train"], 1.0)

        self.assertTrue((self.output_dir / "params.json").exists())
        self.assertTrue((self.output_dir / "output_summary.json").exists())
        self.assertTrue((self.output_dir / "gru_config.json").exists())
        self.assertTrue((self.output_dir / "output_df.csv").exists())

    def test_checkpoint_training(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 4,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 2,
                "checkpoint_plot_split_examples_every_n": 0,
                "checkpoint_save_output_df_every_n": 0,
                "checkpoint_log_eval_to_wandb": False,
                "checkpoint_log_split_examples_to_wandb": False,
                "checkpoint_run_heldout_eval": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.bundle)

        self.assertIn("checkpoints", output)
        self.assertIn("initial_evaluations", output)
        self.assertIn("before_training", output["initial_evaluations"])
        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            self.assertIn("train_likelihood", checkpoint)
            self.assertIsInstance(checkpoint["train_likelihood"], float)
            self.assertGreaterEqual(checkpoint["train_likelihood"], 0.0)
            self.assertLessEqual(checkpoint["train_likelihood"], 1.0)
        self.assertTrue((self.output_dir / "checkpoints" / "index.json").exists())

    def test_early_stop_final_metadata_uses_completed_step(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 4,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 2,
                "checkpoint_plot_split_examples_every_n": 0,
                "checkpoint_save_output_df_every_n": 0,
                "checkpoint_log_eval_to_wandb": False,
                "checkpoint_log_split_examples_to_wandb": False,
                "checkpoint_run_heldout_eval": False,
                "early_stopping": {
                    "enabled": True,
                    "metric": "eval_likelihood",
                    "patience": 0,
                    "start_after_step": 0,
                },
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.bundle)
        index = json.loads((self.output_dir / "checkpoints" / "index.json").read_text())

        self.assertEqual(output["training_steps_completed"], 2)
        self.assertEqual(output["training_steps_requested"], 4)
        self.assertEqual(index["n_steps"], 2)
        self.assertEqual(index["completed_steps"], 2)
        self.assertEqual(index["requested_n_steps"], 4)
        self.assertTrue(index["early_stopped"])
        self.assertTrue((self.output_dir / "checkpoints" / "step_2").exists())
        self.assertFalse((self.output_dir / "checkpoints" / "step_4").exists())

    def test_multisubject_training_exports_subject_artifacts(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 2,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.multisubject_bundle)

        self.assertTrue(output["multisubject"])
        self.assertNotIn("parameter_training", output)
        self.assertIn("subject_artifacts", output)
        self.assertTrue((self.output_dir / "subject_index_map.json").exists())
        self.assertTrue((self.output_dir / "subject_embeddings.pkl").exists())
        self.assertTrue((self.output_dir / "subject_embedding_state_space.png").exists())
        before_training = output["initial_evaluations"]["before_training"]
        self.assertIn("plot_paths", before_training)
        self.assertTrue(before_training["plot_paths"]["subject_embedding_state_space"])
        self.assertTrue(Path(before_training["plot_paths"]["subject_embedding_state_space"]).exists())

        subject_embeddings_df = pd.read_pickle(self.output_dir / "subject_embeddings.pkl")
        self.assertEqual(subject_embeddings_df["subject_id"].tolist(), [711041, 793446])

        params = json.loads((self.output_dir / "params.json").read_text())
        self.assertIn("multisubject_gru", params)
        self.assertIn("subject_embeddings", params["multisubject_gru"])

    def test_freeze_gru_core_updates_only_embeddings_and_readout(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-2,
                "n_steps": 5,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "freeze_gru_core": True,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.multisubject_bundle)
        initial = json.loads(
            (self.output_dir / "initialization" / "before_training" / "params.json").read_text()
        )
        final = json.loads((self.output_dir / "params.json").read_text())

        gru_module = next(name for name in initial if name.endswith("/~/gru"))
        readout_module = next(name for name in initial if name.endswith("/~/readout"))
        for parameter_name, initial_value in initial[gru_module].items():
            self.assertTrue(
                np.array_equal(initial_value, final[gru_module][parameter_name]),
                f"GRU parameter changed: {parameter_name}",
            )
        self.assertFalse(
            np.array_equal(
                initial["multisubject_gru"]["subject_embeddings"],
                final["multisubject_gru"]["subject_embeddings"],
            )
        )
        self.assertTrue(
            any(
                not np.array_equal(initial_value, final[readout_module][parameter_name])
                for parameter_name, initial_value in initial[readout_module].items()
            )
        )
        self.assertTrue(output["parameter_training"]["freeze_gru_core"])
        self.assertGreater(output["parameter_training"]["trainable_parameter_count"], 0)
        self.assertGreater(output["parameter_training"]["frozen_parameter_count"], 0)

    def test_freeze_gru_core_requires_multisubject_mode(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "freeze_gru_core": True,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(ValueError, "requires a multisubject GRU"):
            trainer.fit(self.bundle)

    def test_multisubject_checkpoint_training_plots_subject_embeddings(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 4,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 2,
                "checkpoint_plot_split_examples_every_n": 0,
                "checkpoint_save_output_df_every_n": 0,
                "checkpoint_log_eval_to_wandb": False,
                "checkpoint_log_split_examples_to_wandb": False,
                "checkpoint_keep_media_files": True,
                "checkpoint_run_heldout_eval": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.multisubject_bundle)

        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            plot_paths = checkpoint.get("plot_paths", {})
            self.assertTrue(plot_paths["subject_embedding_state_space"])
            self.assertTrue(Path(plot_paths["subject_embedding_state_space"]).exists())

    def test_multisubject_scalar_session_conditioning_saves_session_context_artifact(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
                "session_encoding_type": "scalar",
                "session_integration_type": "direct",
                "session_delta_n_layers": 2,
                "session_delta_hidden_size": 11,
                "session_n_pretrain_steps": 4,
                "session_n_warmup_steps": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir / "scalar_session"),
            seed=42,
        )

        output = trainer.fit(self.multisubject_bundle)

        subject_artifacts = output["subject_artifacts"]
        self.assertIn("session_context_map", subject_artifacts)
        self.assertTrue(Path(subject_artifacts["session_context_map"]).exists())
        self.assertIn("subject_session_context_state_space_path", output)
        self.assertTrue(Path(output["subject_session_context_state_space_path"]).exists())
        before_training = output["initial_evaluations"]["before_training"]
        self.assertTrue(before_training["plot_paths"]["subject_session_context_state_space"])
        saved_cfg = json.loads((self.output_dir / "scalar_session" / "gru_config.json").read_text())
        self.assertEqual(saved_cfg["architecture"]["session_encoding_type"], "scalar")
        self.assertEqual(saved_cfg["architecture"]["session_integration_type"], "direct")
        self.assertEqual(saved_cfg["architecture"]["session_delta_n_layers"], 2)
        self.assertEqual(saved_cfg["architecture"]["session_delta_hidden_size"], 11)
        self.assertEqual(saved_cfg["architecture"]["session_n_pretrain_steps"], 4)
        self.assertEqual(saved_cfg["architecture"]["session_n_warmup_steps"], 3)

    def test_multisubject_fourier_pre_mlp_session_conditioning_persists_config(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
                "session_encoding_type": "fourier",
                "session_integration_type": "pre_mlp",
                "session_fourier_k": 3,
                "session_delta_n_layers": 2,
                "session_delta_hidden_size": 11,
                "session_n_pretrain_steps": 5,
                "session_n_warmup_steps": 2,
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir / "fourier_session"),
            seed=42,
        )

        trainer.fit(self.multisubject_bundle)

        saved_cfg = json.loads((self.output_dir / "fourier_session" / "gru_config.json").read_text())
        self.assertEqual(saved_cfg["architecture"]["session_encoding_type"], "fourier")
        self.assertEqual(saved_cfg["architecture"]["session_integration_type"], "pre_mlp")
        self.assertEqual(saved_cfg["architecture"]["session_fourier_k"], 3)
        self.assertEqual(saved_cfg["architecture"]["session_delta_n_layers"], 2)
        self.assertEqual(saved_cfg["architecture"]["session_delta_hidden_size"], 11)
        self.assertEqual(saved_cfg["architecture"]["session_n_pretrain_steps"], 5)
        self.assertEqual(saved_cfg["architecture"]["session_n_warmup_steps"], 2)

    def test_multisubject_session_conditioning_resolves_default_curriculum_steps(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
                "session_encoding_type": "scalar",
            },
            training={
                "lr": 1e-3,
                "n_steps": 10,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir / "resolved_curriculum"),
            seed=42,
        )

        trainer.fit(self.multisubject_bundle)

        saved_cfg = json.loads(
            (self.output_dir / "resolved_curriculum" / "gru_config.json").read_text()
        )
        self.assertEqual(saved_cfg["architecture"]["session_n_pretrain_steps"], 3)
        self.assertEqual(saved_cfg["architecture"]["session_n_warmup_steps"], 2)

    def test_multisubject_requires_subject_embedding_size(self):
        trainer = GruTrainer(
            architecture={"multisubject": True, "hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "architecture.subject_embedding_size > 0",
        ):
            trainer.fit(self.multisubject_bundle)

    def test_multisubject_accepts_all_subject_embedding_init_modes(self):
        for init_mode in ("zeros", "small_random", "subject_count_scaled_random"):
            trainer = GruTrainer(
                architecture={
                    "multisubject": True,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "subject_embedding_size": 3,
                    "subject_embedding_init": init_mode,
                },
                training={
                    "lr": 1e-3,
                    "n_steps": 0,
                    "loss": "categorical",
                    "loss_param": 1,
                    "max_grad_norm": 1.0,
                    "checkpoint_run_heldout_eval": False,
                    "save_output_df": False,
                },
                output_dir=str(self.output_dir / init_mode),
                seed=42,
            )

            output = trainer.fit(self.multisubject_bundle)
            self.assertTrue(output["multisubject"])

    def test_multisubject_rejects_unknown_subject_embedding_init_mode(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
                "subject_embedding_init": "not_a_real_mode",
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported subject_embedding_init",
        ):
            trainer.fit(self.multisubject_bundle)

    def test_multisubject_requires_subject_count_metadata(self):
        missing_metadata_bundle = DatasetBundle(
            raw=self.multisubject_bundle.raw.copy(),
            train_set=self.multisubject_bundle.train_set,
            eval_set=self.multisubject_bundle.eval_set,
            metadata={
                "ignore_policy": "exclude",
                "eval_every_n": 2,
                "multisubject": True,
            },
            extras=self.multisubject_bundle.extras,
        )
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "metadata.num_subjects or metadata.subject_ids",
        ):
            trainer.fit(missing_metadata_bundle)

    def test_validate_multisubject_dataset_inputs_allows_padding(self):
        class _DummyDataset:
            def __init__(self, xs):
                self._xs = xs

            def get_all(self):
                return {
                    "xs": self._xs,
                    "ys": np.zeros((self._xs.shape[0], self._xs.shape[1], 1)),
                }

        xs = np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[-1.0, -1.0, -1.0]],
            ]
        )

        _validate_multisubject_dataset_inputs(
            _DummyDataset(xs),
            max_n_subjects=1,
            context="test",
        )

    def test_validate_multisubject_dataset_inputs_rejects_out_of_range_ids(self):
        class _DummyDataset:
            def __init__(self, xs):
                self._xs = xs

            def get_all(self):
                return {
                    "xs": self._xs,
                    "ys": np.zeros((self._xs.shape[0], self._xs.shape[1], 1)),
                }

        xs = np.array([[[2.0, 1.0, 0.0]]])

        with self.assertRaisesRegex(ValueError, "out-of-range subject ids"):
            _validate_multisubject_dataset_inputs(
                _DummyDataset(xs),
                max_n_subjects=2,
                context="test",
            )

    def test_validate_multisubject_dataset_inputs_validates_session_indices(self):
        class _DummyDataset:
            def __init__(self, xs):
                self._xs = xs

            def get_all(self):
                return {
                    "xs": self._xs,
                    "ys": np.zeros((self._xs.shape[0], self._xs.shape[1], 1)),
                }

        xs = np.array([[[0.0, 3.0, 1.0, 0.0]]])

        with self.assertRaisesRegex(ValueError, "out-of-range session ids"):
            _validate_multisubject_dataset_inputs(
                _DummyDataset(xs),
                max_n_subjects=1,
                context="test",
                session_conditioning_enabled=True,
                session_max_index_by_subject_index=[2],
            )

    def test_session_conditioning_requires_multisubject_mode(self):
        trainer = GruTrainer(
            architecture={
                "hidden_size": 8,
                "num_layers": 1,
                "session_encoding_type": "scalar",
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir / "invalid_single_subject"),
            seed=42,
        )

        with self.assertRaisesRegex(ValueError, "requires multisubject mode"):
            trainer.fit(self.bundle)

    def test_session_conditioning_rejects_invalid_fourier_k(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
                "session_encoding_type": "fourier",
                "session_fourier_k": 0,
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir / "invalid_fourier"),
            seed=42,
        )

        with self.assertRaisesRegex(ValueError, "session_fourier_k must be > 0"):
            trainer.fit(self.multisubject_bundle)

    def test_session_delta_regularization_skipped_when_conditioning_disabled(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "lambda_reg_session": 1e-3,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir / "invalid_session_reg"),
            seed=42,
        )

        # Conditioning is disabled (session_encoding_type=none by default), so a
        # positive lambda_reg_session is auto-disabled with a warning rather than
        # raising — this is what lets a sweep cross none x scalar arms safely.
        with self.assertLogs(level="WARNING") as captured:
            trainer.fit(self.multisubject_bundle)
        self.assertTrue(
            any(
                "skipping session-delta regularization" in message
                for message in captured.output
            ),
            captured.output,
        )

    def test_heldout_eval_rejects_multisubject_gru(self):
        hydra_config = OmegaConf.create(
            {
                "data": {
                    "test_subject_ids": [711041],
                    "ignore_policy": "exclude",
                    "heldout_example_sessions_per_subject": 0,
                    "example_max_subjects": 1,
                },
                "model": {
                    "architecture": {
                        "multisubject": True,
                        "hidden_size": 8,
                        "output_size": 2,
                    },
                    "output_dir": str(self.output_dir),
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "not supported for multisubject GRU",
        ):
            evaluate_gru_on_heldout_subjects(hydra_config)

    # --- Phase B/C/D refactor guards ------------------------------------------

    def test_inherits_base_and_sets_model_label(self):
        # Pins the parameterization used by the shared base class (T6).
        self.assertTrue(issubclass(GruTrainer, BaseMultisubjectTrainer))
        self.assertEqual(GruTrainer._MODEL_LABEL, "GRU")
        self.assertEqual(GruTrainer._TRAINER_CONTEXT_NAME, "GruTrainer")

    def test_plot_examples_for_split_dispatches_to_gru_without_params(self):
        # T7: GRU's split-example hook calls the GRU plotter and does NOT forward params.
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )
        sentinel = {"plotting_skipped": True}
        with patch(
            "model_trainers.gru_trainer.plot_gru_examples_for_split",
            return_value=sentinel,
        ) as mock_plot:
            result = trainer._plot_examples_for_split(
                split_name="training",
                output_dir=self.output_dir,
                output_df=None,
                network_states=None,
                yhat_logits=None,
                params={"p": 1},
                sessions_per_subject=1,
                max_subjects_to_plot=1,
                n_action_logits=2,
                wandb_run=None,
                log_scope="Test",
            )
        self.assertIs(result, sentinel)
        mock_plot.assert_called_once()
        self.assertNotIn("params", mock_plot.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
