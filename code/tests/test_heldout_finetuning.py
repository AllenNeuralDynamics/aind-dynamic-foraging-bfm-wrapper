from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import jax
    import optax
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from base.types import DatasetBundle
    from data_loaders import mice as mice_loader
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from disentangled_rnns.library import rnn_utils
    from model_trainers.disrnn_trainer import DisrnnTrainer
    from model_trainers.gru_trainer import GruTrainer
    from models.gru_network import make_gru_network
    from models.session_conditioning import resolve_session_conditioning_from_architecture
    from post_training_analysis.heldout_finetuning import (
        _build_final_target_predictions,
        _heldout_subject_run_name_suffix,
        _maybe_start_wandb_run,
        run_heldout_subject_finetuning_from_config,
    )
    from utils.multisubject import (
        expand_local_multisubject_params,
        extract_subject_embeddings_from_params,
        prepend_session_index_to_multisubject_split_datasets,
    )
    from utils.session_regularized_training import (
        train_network_with_session_regularization,
    )

    HELDOUT_FINETUNE_DEPS_AVAILABLE = True
    HELDOUT_FINETUNE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    HELDOUT_FINETUNE_DEPS_AVAILABLE = False
    HELDOUT_FINETUNE_IMPORT_ERROR = exc


class _FakeWandbRun:
    """Minimal W&B run double recording .log() calls and .summary writes."""

    def __init__(self, start_step: int = 0):
        self.step = int(start_step)
        self.logged: list = []  # list of (payload_dict, step)
        self.summary: dict = {}
        self.name = "training-run"
        self.finished = False

    def log(self, payload, step=None):
        self.logged.append((dict(payload), step))
        if step is not None:
            self.step = max(self.step, int(step))

    def finish(self):
        self.finished = True


@unittest.skipUnless(
    HELDOUT_FINETUNE_DEPS_AVAILABLE,
    f"Held-out fine-tuning dependencies unavailable: {HELDOUT_FINETUNE_IMPORT_ERROR}",
)
class TestHeldoutSubjectFinetuning(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="heldout_finetune_test_"))
        self.output_root = self.tmpdir / "results"
        loader = SyntheticCognitiveAgents(
            task={
                "type": "random_walk",
                "reward_baiting": False,
                "p_min": 0.0,
                "p_max": 1.0,
                "sigma": 0.15,
                "mean": 0,
                "num_trials": 24,
                "seed": 11,
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
                    "learn_rate_rew": 0.4,
                    "learn_rate_unrew": 0.1,
                    "forget_rate_unchosen": 0.03,
                    "softmax_inverse_temperature": 4.0,
                    "biasL": 0.0,
                },
                "agent_params_session_var": {
                    "biasL": {
                        "type": "gaussian",
                        "mean": 0.0,
                        "std": 0.5,
                    }
                },
                "seed": 13,
            },
            num_trials=24,
            num_sessions=9,
            eval_every_n=2,
        )
        synthetic_bundle = loader.load()
        self.subject_ids = [711041, 793446, 812345]
        self.train_subject_ids = self.subject_ids[:2]
        self.heldout_subject_ids = self.subject_ids[2:]
        self.full_snapshot_df = self._assign_subject_ids_to_sessions(
            synthetic_bundle.raw.copy(),
            subject_ids=self.subject_ids,
        )
        self.train_snapshot_df = self.full_snapshot_df[
            self.full_snapshot_df["subject_id"].isin(self.train_subject_ids)
        ].copy()
        self.heldout_snapshot_df = self.full_snapshot_df[
            self.full_snapshot_df["subject_id"].isin(self.heldout_subject_ids)
        ].copy()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _assign_subject_ids_to_sessions(
        self,
        raw_df: pd.DataFrame,
        *,
        subject_ids: list[int],
    ) -> pd.DataFrame:
        session_ids = list(dict.fromkeys(raw_df["ses_idx"].tolist()))
        chunk_size = len(session_ids) // len(subject_ids)
        subject_by_session: dict[str, int] = {}
        for subject_index, subject_id in enumerate(subject_ids):
            start = subject_index * chunk_size
            stop = start + chunk_size
            for session_id in session_ids[start:stop]:
                subject_by_session[str(session_id)] = int(subject_id)

        labeled = raw_df.copy()
        labeled["source_ses_idx"] = labeled["ses_idx"].astype(str)
        labeled["subject_id"] = labeled["source_ses_idx"].map(subject_by_session)
        labeled["ses_idx"] = labeled.apply(
            lambda row: f"{int(row['subject_id'])}__{row['source_ses_idx']}",
            axis=1,
        )
        return labeled

    def _make_training_bundle(
        self,
        *,
        session_conditioning: bool,
    ) -> DatasetBundle:
        metadata = {
            "subject_ids": list(self.train_subject_ids),
            "ignore_policy": "exclude",
            "features": None,
            "eval_every_n": 2,
        }
        bundle = mice_loader._build_multisubject_bundle(
            df=self.train_snapshot_df,
            resolved_subject_ids=self.train_subject_ids,
            ignore_policy="exclude",
            features=None,
            eval_every_n=2,
            batch_size=None,
            batch_mode="single",
            metadata=metadata,
        )
        if not session_conditioning:
            return bundle

        dataset_full, dataset_train, dataset_eval = prepend_session_index_to_multisubject_split_datasets(
            dataset=bundle.extras["dataset"],
            dataset_train=bundle.train_set,
            dataset_eval=bundle.eval_set,
            metadata=bundle.metadata,
        )
        return DatasetBundle(
            raw=bundle.raw,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=dict(bundle.metadata),
            extras={"dataset": dataset_full},
        )

    def _write_source_inputs(
        self,
        model_dir: Path,
        *,
        model_type: str,
        architecture: dict[str, object],
        training: dict[str, object],
        penalties: dict[str, object] | None = None,
    ) -> None:
        cfg = {
            "data": {
                "type": "mice_snapshot",
                "subject_ids": list(self.train_subject_ids),
                "test_subject_ids": list(self.heldout_subject_ids),
                "multisubject": True,
                "mature_only": True,
                "curricula": ["Uncoupled Baiting"],
                "ignore_policy": "exclude",
                "features": None,
                "eval_every_n": 2,
                "batch_size": None,
                "batch_mode": "single",
            },
            "model": {
                "type": model_type,
                "architecture": architecture,
                "training": training,
            },
            "seed": 7,
        }
        if penalties is not None:
            cfg["model"]["penalties"] = penalties
        OmegaConf.save(config=OmegaConf.create(cfg), f=str(model_dir / "inputs.yaml"))

    def _write_common_multisubject_artifacts(
        self,
        outputs_dir: Path,
        *,
        bundle: DatasetBundle,
    ) -> None:
        subject_index_map = {
            "subject_id_to_index": {
                str(subject_id): int(index)
                for subject_id, index in bundle.metadata["subject_id_to_index"].items()
            },
            "index_to_subject_id": {
                str(index): subject_id
                for index, subject_id in bundle.metadata["index_to_subject_id"].items()
            },
        }
        (outputs_dir / "subject_index_map.json").write_text(json.dumps(subject_index_map, indent=2))
        if bundle.metadata.get("session_context") is not None:
            (outputs_dir / "session_context_map.json").write_text(
                json.dumps(bundle.metadata["session_context"], indent=2)
            )

    def _create_gru_source_run(self, *, session_conditioning: bool) -> tuple[Path, DatasetBundle, Any]:
        model_dir = self.tmpdir / ("gru_session" if session_conditioning else "gru_plain")
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        bundle = self._make_training_bundle(session_conditioning=session_conditioning)

        architecture = {
            "multisubject": True,
            "hidden_size": 6,
            "num_layers": 1,
            "subject_embedding_size": 3,
            "subject_embedding_init": "zeros",
            "session_encoding_type": "scalar" if session_conditioning else "none",
            "session_integration_type": "direct",
            "session_fourier_k": 4,
            "session_delta_n_layers": 2,
            "session_delta_hidden_size": 8,
        }
        training = {
            "lr": 1e-3,
            "n_steps": 5,
            "loss": "categorical",
            "loss_param": 1,
            "max_grad_norm": 1.0,
            "plot_session_context_state_space": True,
            "session_context_plot_max_subjects": 2,
        }
        self._write_source_inputs(
            model_dir,
            model_type="gru",
            architecture=architecture,
            training=training,
        )

        session_cfg = resolve_session_conditioning_from_architecture(
            architecture=architecture,
            metadata=bundle.metadata,
            multisubject=True,
            max_n_subjects=int(bundle.metadata["num_subjects"]),
            subject_embedding_size=int(architecture["subject_embedding_size"]),
            context="GRU heldout source test",
        )
        gru_make_network = make_gru_network(
            hidden_size=int(architecture["hidden_size"]),
            output_size=2,
            multisubject=True,
            max_n_subjects=int(bundle.metadata["num_subjects"]),
            subject_embedding_size=int(architecture["subject_embedding_size"]),
            subject_embedding_init=str(architecture["subject_embedding_init"]),
            session_encoding_type=str(session_cfg["session_encoding_type"]),
            session_integration_type=str(session_cfg["session_integration_type"]),
            session_fourier_k=int(session_cfg["session_fourier_k"]),
            session_delta_n_layers=int(session_cfg["session_delta_n_layers"]),
            session_delta_hidden_size=int(session_cfg["session_delta_hidden_size"]),
            session_max_index_by_subject_index=list(
                bundle.metadata.get("session_max_index_by_subject_index") or []
            ),
        )
        params, _, _ = train_network_with_session_regularization(
            lambda session_curriculum_lambda=1.0: gru_make_network(session_curriculum_lambda),
            bundle.train_set,
            bundle.eval_set,
            loss="categorical",
            loss_param=1,
            n_action_logits=2,
            session_regularization_apply=None,
            session_regularization_scale=0.0,
            opt=optax.adam(1e-3),
            random_key=jax.random.PRNGKey(0),
            n_steps=0,
        )
        (outputs_dir / "params.json").write_text(
            json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder)
        )
        (outputs_dir / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": architecture,
                    "training": training,
                    "output_size": 2,
                },
                indent=2,
            )
        )
        self._write_common_multisubject_artifacts(outputs_dir, bundle=bundle)
        return model_dir, bundle, params

    def _create_disrnn_source_run(self, *, session_conditioning: bool) -> tuple[Path, DatasetBundle, Any]:
        model_dir = self.tmpdir / ("disrnn_session" if session_conditioning else "disrnn_plain")
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        bundle = self._make_training_bundle(session_conditioning=session_conditioning)

        architecture = {
            "multisubject": True,
            "latent_size": 4,
            "update_net_n_units_per_layer": 6,
            "update_net_n_layers": 2,
            "choice_net_n_units_per_layer": 4,
            "choice_net_n_layers": 1,
            "activation": "leaky_relu",
            "subject_embedding_size": 3,
            "subject_embedding_init": "zeros",
            "session_encoding_type": "scalar" if session_conditioning else "none",
            "session_integration_type": "direct",
            "session_fourier_k": 4,
            "session_delta_n_layers": 2,
            "session_delta_hidden_size": 8,
            "use_global_subject_bottleneck": True,
        }
        penalties = {
            "latent_penalty": 1e-3,
            "choice_net_latent_penalty": 1e-3,
            "update_net_obs_penalty": 1e-3,
            "update_net_latent_penalty": 1e-3,
            "subject_penalty": 1e-3,
            "update_net_subject_penalty": 1e-3,
            "choice_net_subject_penalty": 1e-3,
        }
        training = {
            "lr": 1e-3,
            "n_steps": 5,
            "loss": "penalized_categorical",
            "loss_param": 1,
            "max_grad_norm": 1.0,
            "plot_session_context_state_space": True,
            "session_context_plot_max_subjects": 2,
        }
        self._write_source_inputs(
            model_dir,
            model_type="disrnn",
            architecture=architecture,
            training=training,
            penalties=penalties,
        )

        trainer = DisrnnTrainer(
            architecture=architecture,
            penalties=penalties,
            training=training,
            output_dir=str(outputs_dir),
            seed=7,
        )
        disrnn_config, _ = trainer._build_network_configs(
            dataset=bundle.extras["dataset"],
            ignore_policy="exclude",
            metadata=bundle.metadata,
        )
        make_network = trainer._make_network_factory(disrnn_config, multisubject=True)
        params, _, _ = train_network_with_session_regularization(
            lambda session_curriculum_lambda=1.0: make_network(session_curriculum_lambda),
            bundle.train_set,
            bundle.eval_set,
            loss="penalized_categorical",
            loss_param=1,
            n_action_logits=2,
            session_regularization_apply=None,
            session_regularization_scale=0.0,
            opt=optax.adam(1e-3),
            random_key=jax.random.PRNGKey(0),
            n_steps=0,
        )
        (outputs_dir / "params.json").write_text(
            json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder)
        )
        (outputs_dir / "disrnn_config.json").write_text(
            json.dumps(asdict(disrnn_config), indent=2)
        )
        self._write_common_multisubject_artifacts(outputs_dir, bundle=bundle)
        return model_dir, bundle, params

    def _make_runner_config(self, *, model_dir: Path) -> dict[str, object]:
        return {
            "source_run": {
                "model_dir": str(model_dir),
                "checkpoint_policy": "best_eval",
            },
            "heldout_subjects": {
                "test_subject_ids": None,
                "test_subject_start": None,
                "test_subject_end": None,
                "mature_only": None,
                "curricula": None,
                "cols_to_retain": None,
            },
            "heldout_finetuning": {
                "n_steps": 2,
                "lr": 1e-2,
                "checkpoint_every_n_steps": 1,
                "batch_size": None,
                "batch_mode": "single",
                "checkpoint_plot_split_examples_every_n": 1,
                "checkpoint_save_output_df_every_n": 1,
                "train_example_sessions_per_subject": 1,
                "eval_example_sessions_per_subject": 1,
                "example_max_subjects": 2,
                "keep_media_files": True,
            },
            "output": {
                "output_root": str(self.output_root),
                "run_name_suffix": None,
            },
            "seed": 17,
        }

    def _make_external_target_bundle(self) -> DatasetBundle:
        subject_id = self.heldout_subject_ids[0]
        target_df = self.heldout_snapshot_df.copy()
        bundle = mice_loader._build_multisubject_bundle(
            df=target_df,
            resolved_subject_ids=[subject_id],
            ignore_policy="exclude",
            features=None,
            eval_every_n=2,
            batch_size=None,
            batch_mode="single",
            metadata={
                "subject_ids": [subject_id],
                "ignore_policy": "exclude",
                "features": None,
                "eval_every_n": 2,
            },
        )
        metadata = dict(bundle.metadata)
        metadata.update(
            {
                "source": "external_bandit_file",
                "dataset_id": "external-test",
                "species": "mouse",
                "split_strategy": "explicit_manifest",
            }
        )
        return DatasetBundle(
            raw=bundle.raw,
            train_set=bundle.train_set,
            eval_set=bundle.eval_set,
            metadata=metadata,
            extras=bundle.extras,
        )

    def test_runner_accepts_external_target_and_exports_fixed_test_rows(self) -> None:
        model_dir, _, _ = self._create_gru_source_run(session_conditioning=False)
        config = self._make_runner_config(model_dir=model_dir)
        config["heldout_finetuning"]["n_steps"] = 1
        config["heldout_finetuning"]["adapt_sessions_per_subject"] = 1
        result = run_heldout_subject_finetuning_from_config(
            config,
            target_bundle=self._make_external_target_bundle(),
        )

        predictions = pd.read_csv(result["test_trial_predictions_path"])
        metrics = json.loads(Path(result["test_metrics_path"]).read_text())
        target = self._make_external_target_bundle()
        expected_sessions = set(target.metadata["eval_session_ids"])
        self.assertEqual(set(predictions["ses_idx"]), expected_sessions)
        self.assertEqual(metrics["n_trials"], len(predictions))
        self.assertIn("brier_score", metrics)
        self.assertIn("accuracy", metrics)
        summary = json.loads(Path(result["summary_path"]).read_text())
        self.assertEqual(summary["selection_policy"], "fixed_final")
        self.assertFalse(summary["target_test_used_for_selection"])

    def test_external_target_with_null_dataset_id_uses_fallback_slug(self) -> None:
        model_dir, _, _ = self._create_gru_source_run(session_conditioning=False)
        config = self._make_runner_config(model_dir=model_dir)
        config["heldout_finetuning"]["n_steps"] = 1
        target = self._make_external_target_bundle()
        target = DatasetBundle(
            raw=target.raw,
            train_set=target.train_set,
            eval_set=target.eval_set,
            metadata={**target.metadata, "dataset_id": None},
            extras=target.extras,
        )

        result = run_heldout_subject_finetuning_from_config(
            config,
            target_bundle=target,
        )

        self.assertIn("target_external_target", Path(result["output_dir"]).name)

    def test_final_target_predictions_skips_nonbinary_ignore_policy(self) -> None:
        target = self._make_external_target_bundle()
        target = DatasetBundle(
            raw=target.raw,
            train_set=target.train_set,
            eval_set=target.eval_set,
            metadata={**target.metadata, "ignore_policy": "include"},
            extras=target.extras,
        )

        predictions, metrics = _build_final_target_predictions(
            model_type="gru",
            params=None,
            make_eval_network=None,
            bundle=target,
        )

        self.assertIsNone(predictions)
        self.assertIsNone(metrics)

    def test_external_target_zero_shot_keeps_mean_initialized_embedding(self) -> None:
        model_dir, _, source_params = self._create_gru_source_run(
            session_conditioning=False
        )
        config = self._make_runner_config(model_dir=model_dir)
        config["heldout_finetuning"]["adapt_sessions_per_subject"] = 0
        result = run_heldout_subject_finetuning_from_config(
            config,
            target_bundle=self._make_external_target_bundle(),
        )

        checkpoint_metrics = json.loads(Path(result["checkpoint_metrics_path"]).read_text())
        self.assertEqual([record["step"] for record in checkpoint_metrics], [0])
        final_params = rnn_utils.to_np(json.loads(Path(result["params_path"]).read_text()))
        expected_params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=1,
            init="mean",
        )
        np.testing.assert_allclose(
            extract_subject_embeddings_from_params(final_params),
            extract_subject_embeddings_from_params(expected_params),
        )

    def test_gru_runner_finetunes_only_new_subject_rows(self) -> None:
        model_dir, _, source_params = self._create_gru_source_run(session_conditioning=False)
        config = self._make_runner_config(model_dir=model_dir)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            return_value=(self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)),
        ):
            result = run_heldout_subject_finetuning_from_config(config)

        checkpoint_metrics = json.loads(Path(result["checkpoint_metrics_path"]).read_text())
        self.assertEqual([record["step"] for record in checkpoint_metrics], [0, 1, 2])
        self.assertIn("train_loss", checkpoint_metrics[0])
        self.assertIn("eval_likelihood", checkpoint_metrics[-1])
        self.assertTrue(Path(result["loss_curve_path"]).exists())
        self.assertTrue(Path(result["likelihood_curve_path"]).exists())

        final_params = rnn_utils.to_np(json.loads(Path(result["params_path"]).read_text()))
        initial_expanded_params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=1,
            init="mean",
        )
        initial_embeddings = extract_subject_embeddings_from_params(initial_expanded_params)
        final_embeddings = extract_subject_embeddings_from_params(final_params)
        self.assertEqual(final_embeddings.shape[0], 3)
        self.assertTrue(np.allclose(final_embeddings[:2], initial_embeddings[:2]))
        self.assertFalse(np.allclose(final_embeddings[2], initial_embeddings[2]))

        subject_map_path = Path(result["subject_index_map_path"])
        self.assertTrue(subject_map_path.exists())
        subject_map = json.loads(subject_map_path.read_text())
        self.assertIn(str(self.heldout_subject_ids[0]), subject_map["subject_id_to_index"])
        self.assertFalse(
            (Path(result["outputs_dir"]) / "session_context_map.json").exists()
        )

    def test_disrnn_runner_supports_session_conditioning(self) -> None:
        model_dir, _, source_params = self._create_disrnn_source_run(session_conditioning=True)
        config = self._make_runner_config(model_dir=model_dir)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            return_value=(self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)),
        ):
            result = run_heldout_subject_finetuning_from_config(config)

        outputs_dir = Path(result["outputs_dir"])
        checkpoint_metrics = json.loads(Path(result["checkpoint_metrics_path"]).read_text())
        self.assertEqual(len(checkpoint_metrics), 3)
        self.assertTrue((outputs_dir / "session_context_map.json").exists())
        self.assertTrue((outputs_dir / "checkpoints" / "step_0" / "params.json").exists())
        self.assertTrue((outputs_dir / "checkpoints" / "step_2" / "output_df.csv").exists())

        disrnn_config = json.loads(Path(result["model_config_path"]).read_text())
        self.assertEqual(int(disrnn_config["max_n_subjects"]), 3)
        self.assertEqual(len(disrnn_config["session_max_index_by_subject_index"]), 3)

        final_params = rnn_utils.to_np(json.loads(Path(result["params_path"]).read_text()))
        initial_expanded_params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=1,
            init="mean",
        )
        initial_embeddings = extract_subject_embeddings_from_params(initial_expanded_params)
        final_embeddings = extract_subject_embeddings_from_params(final_params)
        self.assertTrue(np.allclose(final_embeddings[:2], initial_embeddings[:2]))
        self.assertFalse(np.allclose(final_embeddings[2], initial_embeddings[2]))

    def test_runner_falls_back_to_source_run_heldout_selectors(self) -> None:
        # With no explicit selectors, the heldout selector is derived from the
        # source run's pipeline params (the reserved every-Nth subjects) — no error.
        from post_training_analysis.heldout_finetuning import _resolve_heldout_selector

        selector = _resolve_heldout_selector(
            config={"heldout_subjects": {}},
            source_data_cfg={
                "curricula": ["Coupled Baiting"],
                "min_sessions": 12,
                "heldout_every_n": 4,
                "mature_only": True,
            },
        )
        self.assertIsNone(selector["test_subject_ids"])
        self.assertEqual(selector["curricula"], ["Coupled Baiting"])
        self.assertEqual(selector["min_sessions"], 12)
        self.assertEqual(selector["heldout_every_n"], 4)

    def test_runner_allows_config_override_for_heldout_selectors(self) -> None:
        model_dir, _, _ = self._create_gru_source_run(session_conditioning=False)
        cfg_path = model_dir / "inputs.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.data.test_subject_ids = None
        OmegaConf.save(cfg, f=str(cfg_path))

        config = self._make_runner_config(model_dir=model_dir)
        config["heldout_subjects"] = {
            "test_subject_ids": list(self.heldout_subject_ids),
            "test_subject_start": None,
            "test_subject_end": None,
            "mature_only": True,
            "curricula": ["Uncoupled Baiting"],
            "cols_to_retain": None,
        }

        captured_selector: dict[str, object] = {}

        def _fake_load_selector(*, heldout_selector):
            captured_selector.update(dict(heldout_selector))
            return self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            side_effect=_fake_load_selector,
        ):
            run_heldout_subject_finetuning_from_config(config, output_root=self.output_root)

        self.assertEqual(
            captured_selector["test_subject_ids"],
            list(self.heldout_subject_ids),
        )

    def test_heldout_subject_run_name_suffix_is_compact_and_stable(self):
        suffix = _heldout_subject_run_name_suffix(["s1", "s2", "s3"])

        self.assertRegex(suffix, r"^heldout-n3-[0-9a-f]{8}$")
        self.assertEqual(suffix, _heldout_subject_run_name_suffix(["s1", "s2", "s3"]))
        self.assertNotIn("s1-s2-s3", suffix)
        self.assertEqual(_heldout_subject_run_name_suffix([]), "")

    def test_standalone_wandb_init_uses_retry_fallback_helper(self):
        fake_run = _FakeWandbRun()
        resolved_config = {"wandb": {"project": "proj"}, "meta": {"kind": "test"}}
        source_run = SimpleNamespace(
            model_type="gru",
            run_config={"wandb": {"name": "source-run"}},
        )

        with mock.patch(
            "utils.run_helpers._init_wandb_with_fallback",
            return_value=fake_run,
        ) as init_with_fallback:
            run = _maybe_start_wandb_run(
                resolved_config=resolved_config,
                run_dir=self.output_root,
                source_run=source_run,
            )

        self.assertIs(run, fake_run)
        init_with_fallback.assert_called_once()
        init_kwargs = init_with_fallback.call_args.args[0]
        self.assertEqual(init_kwargs["project"], "proj")
        self.assertEqual(init_kwargs["name"], "source-run")
        self.assertEqual(init_kwargs["config"], resolved_config)
        self.assertIn("heldout_subject_finetuning", init_kwargs["tags"])
        self.assertIn("gru", init_kwargs["tags"])

    # --- Auto held-out fine-tuning: injected W&B run + namespacing ------------

    def test_injected_wandb_run_logs_heldout_namespace_with_step_offset(self):
        # Test A: when a W&B run is injected, held-out train/eval likelihoods are
        # logged into it under `heldout/`, at offset (non-colliding) steps, and the
        # injected run is neither renamed nor finished.
        model_dir, _, _ = self._create_gru_source_run(session_conditioning=False)
        config = self._make_runner_config(model_dir=model_dir)
        fake_run = _FakeWandbRun(start_step=42)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            return_value=(self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)),
        ):
            run_heldout_subject_finetuning_from_config(
                config,
                wandb_run=fake_run,
                wandb_key_prefix="heldout",
                wandb_step_offset=100,
            )

        scalar_logs = [
            (payload, step)
            for payload, step in fake_run.logged
            if "heldout/train_likelihood" in payload
        ]
        self.assertTrue(scalar_logs, "expected heldout scalar likelihood logs")
        for payload, _ in scalar_logs:
            self.assertIn("heldout/eval_likelihood", payload)
            self.assertIn("heldout/train_loss", payload)
        # checkpoint_every_n_steps=1, n_steps=2 -> fine-tune steps 0,1,2 at offset 100.
        self.assertEqual([step for _, step in scalar_logs], [100, 101, 102])
        # All logged steps respect the offset and are non-decreasing.
        all_steps = [step for _, step in fake_run.logged if step is not None]
        self.assertTrue(all(step >= 100 for step in all_steps))
        self.assertEqual(all_steps, sorted(all_steps))
        # No training-namespace keys leaked from the held-out phase.
        for payload, _ in fake_run.logged:
            self.assertNotIn("checkpoint/train_likelihood", payload)
        # Final aggregate likelihoods recorded under heldout/final/*.
        self.assertIn("heldout/final/train_likelihood", fake_run.summary)
        self.assertIn("heldout/final/eval_likelihood", fake_run.summary)
        # Loss/likelihood-over-checkpoints curves are logged as images (parity with
        # the training process's fig/validation_loss_curve).
        logged_keys = {key for payload, _ in fake_run.logged for key in payload}
        self.assertIn("heldout/fig/loss_curve", logged_keys)
        self.assertIn("heldout/fig/likelihood_curve", logged_keys)
        # The injected (training) run must not be renamed or finished here.
        self.assertEqual(fake_run.name, "training-run")
        self.assertFalse(fake_run.finished)
        self.assertNotIn("output_dir", fake_run.summary)
        # split digests of the held-out cohort's adapt / scored sessions (wrapper #131)
        self.assertNotIn("split/error", fake_run.summary)
        for role in ("heldout_context", "heldout_eval"):
            self.assertRegex(fake_run.summary[f"split/{role}/sessions_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(fake_run.summary[f"split/{role}/n_subjects"],
                             len(self.heldout_subject_ids))

    def test_injected_run_defaults_preserve_checkpoint_namespace(self):
        # Test B: defaults (no prefix/offset) reproduce the legacy `checkpoint/*`
        # keys and un-offset steps even when an external run is injected.
        model_dir, _, _ = self._create_gru_source_run(session_conditioning=False)
        config = self._make_runner_config(model_dir=model_dir)
        fake_run = _FakeWandbRun()

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            return_value=(self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)),
        ):
            run_heldout_subject_finetuning_from_config(config, wandb_run=fake_run)

        scalar_logs = [
            (payload, step)
            for payload, step in fake_run.logged
            if "checkpoint/train_likelihood" in payload
        ]
        self.assertTrue(scalar_logs)
        self.assertEqual([step for _, step in scalar_logs], [0, 1, 2])
        self.assertIn("checkpoint/final/train_likelihood", fake_run.summary)
        self.assertFalse(fake_run.finished)

    # --- training_runner auto-hook gating + config synthesis -----------------

    def test_run_capsule_auto_finetune_gating_and_synthesis(self):
        # Test C: the training_runner helpers gate on the config flag and synthesize a
        # finetune config pointing at /results with the heldout namespace.
        import training_runner

        def _cfg(enabled):
            return OmegaConf.create(
                {
                    "seed": 7,
                    "model": {
                        "type": "gru",
                        "training": {
                            "auto_heldout_finetune": {
                                "enabled": enabled,
                                "n_steps": 33,
                                "lr": 5e-3,
                                "checkpoint_policy": "final",
                            }
                        },
                    },
                }
            )

        self.assertTrue(training_runner._auto_heldout_finetune_enabled(_cfg(True)))
        self.assertFalse(training_runner._auto_heldout_finetune_enabled(_cfg(False)))
        self.assertFalse(
            training_runner._auto_heldout_finetune_enabled(
                OmegaConf.create({"model": {"type": "gru", "training": {}}})
            )
        )

        captured = {}

        def _fake_finetune(config, *, wandb_run=None, wandb_key_prefix=None, wandb_step_offset=None):
            captured["config"] = config
            captured["wandb_key_prefix"] = wandb_key_prefix
            captured["wandb_step_offset"] = wandb_step_offset
            return {"ok": True}

        fake_run = _FakeWandbRun(start_step=200)
        with mock.patch.object(
            training_runner, "run_heldout_subject_finetuning_from_config", _fake_finetune
        ):
            result = training_runner._run_auto_heldout_finetune(
                _cfg(True), model_type="gru", wandb_run=fake_run
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["wandb_key_prefix"], "heldout")
        self.assertEqual(captured["wandb_step_offset"], 201)  # wandb_run.step + 1
        self.assertEqual(captured["config"]["source_run"]["model_dir"], "/results")
        self.assertEqual(captured["config"]["source_run"]["checkpoint_policy"], "final")
        self.assertEqual(captured["config"]["heldout_finetuning"]["n_steps"], 33)
        self.assertEqual(captured["config"]["heldout_finetuning"]["lr"], 5e-3)
        # heldout selectors all None -> inherit reserved set from the source run.
        self.assertTrue(
            all(value is None for value in captured["config"]["heldout_subjects"].values())
        )
        self.assertNotIn("wandb", captured["config"])
