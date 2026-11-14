"""Tests for post_training_analysis.heldout_rescore (wrapper #126).

Hand-set data: random left/right/ignore choices for a few mice, and a GRU whose params are
only initialised (no training), so the tests check the bookkeeping, not model quality.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import jax
    import numpy as np
    import optax
    import pandas as pd
    import yaml

    from data_loaders import mice as mice_loader
    from disentangled_rnns.library import rnn_utils
    from models.gru_network import make_gru_network
    from post_training_analysis import heldout_rescore
    from post_training_analysis.generative_analysis import resolve_model_run
    from post_training_analysis.heldout_finetuning import (
        run_heldout_subject_finetuning_from_config,
    )
    from resume_heldout_beaker import _build_finetune_config
    from utils.multisubject import (
        compute_train_eval_session_ids,
        prepend_session_index_to_multisubject_split_datasets,
    )
    from utils.session_regularized_training import train_network_with_session_regularization

    DEPS_AVAILABLE = True
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    DEPS_AVAILABLE = False
    IMPORT_ERROR = exc

TRAIN_SUBJECTS = ["700001", "700002"]
HELDOUT_SUBJECTS = ["800001", "800002"]
N_SESSIONS = 6
N_TRIALS = 30
LOAD_PATCH = "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection"


def _trials(subjects, *, seed: int) -> "pd.DataFrame":
    """Random choices with ~10% ignores; session ids unique across mice, like the database."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in subjects:
        for session in range(N_SESSIONS):
            response = rng.choice([0, 1, 2], size=N_TRIALS, p=[0.45, 0.45, 0.10])
            reward = (rng.random(N_TRIALS) < 0.4) & (response != 2)
            for trial in range(N_TRIALS):
                rows.append(
                    {
                        "subject_id": subject,
                        "ses_idx": f"{subject}_2026-01-{session + 10:02d}_120000",
                        "trial": trial,
                        "animal_response": int(response[trial]),
                        "earned_reward": float(reward[trial]),
                    }
                )
    return pd.DataFrame(rows)


@unittest.skipUnless(DEPS_AVAILABLE, f"dependencies unavailable: {IMPORT_ERROR}")
class TestReconcileSessionRoles(unittest.TestCase):
    def setUp(self):
        self.loader = pd.DataFrame(
            {
                "subject_id": ["a", "a", "a", "a", "b", "b"],
                "session_id": ["a1", "a2", "a3", "a4", "b1", "b2"],
            }
        )
        self.roles = pd.DataFrame(
            {
                "subject_id": ["a", "a", "a", "b", "b", "b", "z"],
                "session_id": ["a1", "a2", "a3", "b1", "b2", "b9", "z1"],
                "role": ["fit", "score", "fit", "score", "fit", "fit", "fit"],
            }
        )

    def test_pass_a_and_swap_select_exactly_the_listed_sessions(self):
        split_a, report = heldout_rescore.reconcile_session_roles(
            self.loader, self.roles, subjects=["a", "b"], swap=False
        )
        self.assertEqual(split_a["a"], {"adapt_session_ids": ["a1", "a3"], "test_session_ids": ["a2"]})
        self.assertEqual(split_a["b"], {"adapt_session_ids": ["b2"], "test_session_ids": ["b1"]})
        split_b, _ = heldout_rescore.reconcile_session_roles(
            self.loader, self.roles, subjects=["a", "b"], swap=True
        )
        self.assertEqual(split_b["a"], {"adapt_session_ids": ["a2"], "test_session_ids": ["a1", "a3"]})
        # Mismatches are counted and listed; mouse z is not requested, so its row is ignored.
        self.assertEqual(
            report["sessions_in_loader_not_in_table"], {"count": 1, "session_ids": ["a4"]}
        )
        self.assertEqual(
            report["sessions_in_table_not_in_loader"], {"count": 1, "session_ids": ["b9"]}
        )
        self.assertEqual(report["n_sessions_used"], 5)

    def test_mouse_without_a_scored_session_is_an_error(self):
        roles = self.roles.copy()
        roles.loc[roles["session_id"] == "a2", "role"] = "fit"
        with self.assertRaisesRegex(ValueError, r"\['a'\]"):
            heldout_rescore.reconcile_session_roles(self.loader, roles, subjects=["a"], swap=False)

    def test_mouse_without_matched_sessions_is_an_error(self):
        # z has a role row but no loader session; c has neither.
        for subjects in (["a", "z"], ["a", "c"]):
            with self.assertRaisesRegex(ValueError, "or any session"):
                heldout_rescore.reconcile_session_roles(
                    self.loader, self.roles, subjects=subjects, swap=False
                )

    def test_role_table_rejects_unknown_roles_and_duplicates(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "roles.csv"
            self.roles.assign(role="train").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "roles"):
                heldout_rescore.read_role_table(path)
            pd.concat([self.roles, self.roles.head(1)]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "more than once"):
                heldout_rescore.read_role_table(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(DEPS_AVAILABLE, f"dependencies unavailable: {IMPORT_ERROR}")
class TestHeldoutRescore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="heldout_rescore_test_"))
        self.heldout_df = _trials(HELDOUT_SUBJECTS, seed=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _source_run(self, ignore_policy: str, session_conditioning: bool = False) -> Path:
        """A source run dir with two checkpoints: step 10 (best eval, lambda 0) and step 20."""
        model_dir = self.tmp / f"source_{ignore_policy}{'_session' if session_conditioning else ''}"
        outputs = model_dir / "outputs"
        outputs.mkdir(parents=True)
        bundle = mice_loader._build_multisubject_bundle(
            df=_trials(TRAIN_SUBJECTS, seed=1),
            resolved_subject_ids=TRAIN_SUBJECTS,
            ignore_policy=ignore_policy,
            features=None,
            eval_every_n=2,
            batch_size=None,
            batch_mode="single",
            metadata={"ignore_policy": ignore_policy, "eval_every_n": 2},
        )
        output_size = 3 if ignore_policy == "include" else 2
        architecture = {
            "multisubject": True,
            "hidden_size": 5,
            "num_layers": 1,
            "subject_embedding_size": 2,
            "subject_embedding_init": "zeros",
            "session_encoding_type": "scalar" if session_conditioning else "none",
            "session_integration_type": "direct",
            "session_fourier_k": 4,
            "session_delta_n_layers": 2,
            "session_delta_hidden_size": 4,
        }
        network = make_gru_network(
            hidden_size=5,
            output_size=output_size,
            multisubject=True,
            max_n_subjects=len(TRAIN_SUBJECTS),
            subject_embedding_size=2,
            session_encoding_type=architecture["session_encoding_type"],
            session_delta_n_layers=2,
            session_delta_hidden_size=4,
            session_max_index_by_subject_index=bundle.metadata["session_max_index_by_subject_index"],
        )
        if session_conditioning:
            _, train_set, eval_set = prepend_session_index_to_multisubject_split_datasets(
                dataset=bundle.extras["dataset"],
                dataset_train=bundle.train_set,
                dataset_eval=bundle.eval_set,
                metadata=bundle.metadata,
            )
            bundle = bundle.__class__(
                raw=bundle.raw, train_set=train_set, eval_set=eval_set,
                metadata=bundle.metadata, extras=bundle.extras,
            )
        params, _, _ = train_network_with_session_regularization(
            lambda session_curriculum_lambda=1.0: network(session_curriculum_lambda),
            bundle.train_set,
            bundle.eval_set,
            loss="categorical",
            loss_param=1,
            n_action_logits=output_size,
            session_regularization_apply=None,
            session_regularization_scale=0.0,
            opt=optax.adam(1e-3),
            random_key=jax.random.PRNGKey(0),
            n_steps=0,
        )
        params_text = json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder)
        (outputs / "params.json").write_text(params_text)
        checkpoints = []
        for step, likelihood, lam in ((10, 0.6, 0.0), (20, 0.5, 1.0)):
            (outputs / "checkpoints" / f"step_{step}").mkdir(parents=True)
            (outputs / "checkpoints" / f"step_{step}" / "params.json").write_text(params_text)
            checkpoints.append(
                {
                    "step": step,
                    "params_path": f"outputs/checkpoints/step_{step}/params.json",
                    "eval_likelihood": likelihood,
                    "session_curriculum_lambda": lam,
                }
            )
        (outputs / "checkpoints" / "index.json").write_text(
            json.dumps({"n_steps": 20, "checkpoints": checkpoints})
        )
        (outputs / "gru_config.json").write_text(
            json.dumps({"architecture": architecture, "output_size": output_size})
        )
        (outputs / "subject_index_map.json").write_text(
            json.dumps(
                {
                    "subject_id_to_index": dict(bundle.metadata["subject_id_to_index"]),
                    "index_to_subject_id": {
                        str(k): v for k, v in bundle.metadata["index_to_subject_id"].items()
                    },
                }
            )
        )
        (outputs / "session_context_map.json").write_text(
            json.dumps(bundle.metadata["session_context"])
        )
        inputs = {
            "data": {
                "type": "mice_snapshot",
                "multisubject": True,
                "mature_only": False,
                "ignore_policy": ignore_policy,
                "features": None,
                "eval_every_n": 2,
                "batch_size": None,
                "test_subject_ids": None,
            },
            "model": {
                "type": "gru",
                "architecture": architecture,
                "training": {
                    "loss": "categorical",
                    "max_grad_norm": 1.0,
                    "auto_heldout_finetune": {
                        "n_steps": 3,
                        "lr": 1e-2,
                        "checkpoint_every_n_steps": 1,
                        "batch_mode": "single",
                        "checkpoint_plot_split_examples_every_n": 0,
                    },
                },
            },
            "seed": None,
        }
        (model_dir / "inputs.yaml").write_text(yaml.safe_dump(inputs))
        return model_dir

    def _alternation_roles(self, ignore_policy: str) -> Path:
        """The eval_every_n=2 alternation the existing held-out fine-tune uses, as a table."""
        aligned = mice_loader._align_raw_df_with_valid_sessions(
            self.heldout_df, ignore_policy=ignore_policy
        )
        rows = []
        for subject, frame in aligned.groupby("subject_id"):
            fit, score = compute_train_eval_session_ids(sorted(frame["ses_idx"].unique()), 2)
            rows += [(subject, s, "fit") for s in fit] + [(subject, s, "score") for s in score]
        path = self.tmp / f"roles_{ignore_policy}.csv"
        pd.DataFrame(rows, columns=["subject_id", "session_id", "role"]).to_csv(path, index=False)
        return path

    def _rescore(self, model_dir: Path, roles: Path, *extra: str) -> tuple[Path, dict]:
        out = self.tmp / f"rescore_{len(list(self.tmp.iterdir()))}"
        with mock.patch(LOAD_PATCH, return_value=(self.heldout_df.copy(), list(HELDOUT_SUBJECTS))):
            manifest = heldout_rescore.main(
                [
                    "--model-dir", str(model_dir),
                    "--subjects", ",".join(HELDOUT_SUBJECTS),
                    "--roles", str(roles),
                    "--output-dir", str(out),
                    *extra,
                ]
            )
        return out, manifest

    def test_alternation_roles_reproduce_existing_heldout_finetune(self):
        """Equivalence: a role table equal to eval_every_n gives the existing per-mouse scores."""
        model_dir = self._source_run("include")
        config = _build_finetune_config(
            yaml.safe_load((model_dir / "inputs.yaml").read_text()), model_dir
        )
        config["output"]["output_root"] = str(self.tmp / "existing")
        with mock.patch(LOAD_PATCH, return_value=(self.heldout_df.copy(), list(HELDOUT_SUBJECTS))):
            existing = run_heldout_subject_finetuning_from_config(config)
        self.assertIsNone(existing["trial_predictions_path"])  # default path unchanged
        records = json.loads(
            (Path(existing["outputs_dir"]) / "per_subject_likelihood.json").read_text()
        )["records"]
        expected = {r["heldout_subject_id"]: r["likelihood"] for r in records if r["ses_idx"] is None}

        out, manifest = self._rescore(
            model_dir, self._alternation_roles("include"), "--session-lambda", "1"
        )
        mice = pd.read_csv(out / "mouse_scores.csv", dtype={"subject_id": str})
        scored = mice[mice["use"] == "scored"].set_index("subject_id")
        self.assertEqual(sorted(scored.index), sorted(expected))
        for subject, likelihood in expected.items():
            row = scored.loc[subject]
            self.assertAlmostEqual(
                float(np.exp(row["ll_3way_sum"] / row["n_trials_3way"])), likelihood, places=6
            )
        self.assertEqual(manifest["reconciliation"]["sessions_in_loader_not_in_table"]["count"], 0)

    def test_swap_scores_the_fit_sessions_and_exports_three_way(self):
        model_dir = self._source_run("include")
        roles_path = self._alternation_roles("include")
        roles = pd.read_csv(roles_path, dtype=str)
        out_a, manifest = self._rescore(model_dir, roles_path)
        out_b, manifest_b = self._rescore(model_dir, roles_path, "--swap")
        for out, scored_role in ((out_a, "score"), (out_b, "fit")):
            trials = pd.read_csv(out / "trial_predictions.csv.gz", dtype={"subject_id": str})
            scored = trials[trials["use"] == "scored"]
            self.assertEqual(
                set(scored["session_id"]),
                set(roles.loc[roles["role"] == scored_role, "session_id"]),
            )
            self.assertTrue((scored["role"] == scored_role).all())
            self.assertEqual(len(trials), len(self.heldout_df))  # 3-way: every trial
            np.testing.assert_allclose(trials[["p_left", "p_right", "p_ignore"]].sum(axis=1), 1.0, rtol=1e-5)
            self.assertTrue(trials["ll_lr"].isna().eq(trials["response"] == 2).all())
            embeddings = pd.read_csv(out / "subject_embeddings.csv", dtype={"subject_id": str})
            self.assertEqual(sorted(embeddings["subject_id"]), sorted(HELDOUT_SUBJECTS))
        self.assertEqual(manifest_b["pass"], "B")
        # best_eval picked step 10, whose lambda is 0; the scoring lambda (1) is recorded next to it.
        self.assertEqual(manifest["checkpoint"]["step"], 10)
        self.assertEqual(manifest["checkpoint"]["session_curriculum_lambda"], 0.0)
        # Default --session-lambda checkpoint: the fine-tune and scoring use that lambda.
        self.assertEqual(manifest["checkpoint"]["session_lambda_mode"], "checkpoint")
        self.assertEqual(manifest["checkpoint"]["finetune_session_curriculum_lambda"], 0.0)
        trials = pd.read_csv(out_a / "trial_predictions.csv.gz")
        self.assertTrue((trials["session_lambda"] == 0.0).all())
        self.assertTrue((trials["checkpoint_lambda"] == 0.0).all())

    def test_session_lambda_zero_makes_the_session_delta_inert(self):
        """At the checkpoint's lambda (0) perturbing the session-delta params changes nothing;
        at lambda 1 it does, and 1 reproduces the auto fine-tune's lambda."""
        model_dir = self._source_run("include", session_conditioning=True)
        roles = self._alternation_roles("include")
        runs = {}
        for perturbed in (False, True):
            if perturbed:
                for step in (10, 20):
                    path = model_dir / "outputs" / "checkpoints" / f"step_{step}" / "params.json"
                    params = json.loads(path.read_text())
                    session_keys = [key for key in params if "session" in key]
                    self.assertTrue(session_keys)
                    for key in session_keys:
                        params[key] = {
                            name: (np.asarray(value) + 0.7).tolist()
                            for name, value in params[key].items()
                        }
                    path.write_text(json.dumps(params))
            for mode in ("checkpoint", "1"):
                out, manifest = self._rescore(model_dir, roles, "--session-lambda", mode)
                runs[(perturbed, mode)] = pd.read_csv(out / "trial_predictions.csv.gz")
                if not perturbed and mode == "1":
                    # lambda 1 reproduces the auto held-out fine-tune on this model.
                    config = _build_finetune_config(
                        yaml.safe_load((model_dir / "inputs.yaml").read_text()), model_dir
                    )
                    config["output"]["output_root"] = str(self.tmp / "existing_session")
                    with mock.patch(
                        LOAD_PATCH, return_value=(self.heldout_df.copy(), list(HELDOUT_SUBJECTS))
                    ):
                        existing = run_heldout_subject_finetuning_from_config(config)
                    summary = json.loads(Path(existing["summary_path"]).read_text())
                    scored = runs[(perturbed, mode)].query("use == 'scored'")
                    self.assertAlmostEqual(
                        float(np.exp(scored["ll_3way"].mean())),
                        summary["metrics"]["eval_likelihood"],
                        places=5,
                    )
                expected = 0.0 if mode == "checkpoint" else 1.0
                self.assertEqual(manifest["checkpoint"]["finetune_session_curriculum_lambda"], expected)
                self.assertTrue((runs[(perturbed, mode)]["session_lambda"] == expected).all())
        columns = ["p_left", "p_right", "p_ignore"]
        np.testing.assert_allclose(
            runs[(False, "checkpoint")][columns], runs[(True, "checkpoint")][columns], rtol=1e-6
        )
        self.assertFalse(
            np.allclose(runs[(False, "1")][columns], runs[(True, "1")][columns], rtol=1e-6)
        )

    def test_sessions_missing_from_either_side_are_reported_and_unused(self):
        model_dir = self._source_run("include")
        roles = pd.read_csv(self._alternation_roles("include"), dtype=str)
        dropped = roles["session_id"].iloc[0]
        roles = roles.iloc[1:]
        roles.loc[len(roles) + 1] = [HELDOUT_SUBJECTS[0], "not_in_loader", "score"]
        path = self.tmp / "roles_edited.csv"
        roles.to_csv(path, index=False)
        out, manifest = self._rescore(model_dir, path, "--checkpoint", "step_20")
        report = manifest["reconciliation"]
        self.assertEqual(report["sessions_in_loader_not_in_table"]["session_ids"], [dropped])
        self.assertEqual(report["sessions_in_table_not_in_loader"]["session_ids"], ["not_in_loader"])
        trials = pd.read_csv(out / "trial_predictions.csv.gz")
        self.assertNotIn(dropped, set(trials["session_id"]))
        self.assertEqual(manifest["checkpoint"]["step"], 20)
        self.assertEqual(manifest["checkpoint"]["session_curriculum_lambda"], 1.0)

    def test_source_training_mice_are_rejected(self):
        model_dir = self._source_run("include")
        with self.assertRaisesRegex(ValueError, TRAIN_SUBJECTS[0]):
            heldout_rescore.main(
                [
                    "--model-dir", str(model_dir),
                    "--subjects", ",".join([HELDOUT_SUBJECTS[0], TRAIN_SUBJECTS[0]]),
                    "--roles", str(self._alternation_roles("include")),
                    "--output-dir", str(self.tmp / "rejected"),
                ]
            )

    def test_trial_export_labels_within_session_prefix_suffix_split(self):
        """A prefix/suffix target: only the suffix trials are eval, and the export check holds."""
        model_dir = self._source_run("include")
        subject = HELDOUT_SUBJECTS[0]
        target_df = self.heldout_df[self.heldout_df["subject_id"] == subject]
        session = sorted(target_df["ses_idx"].unique())[0]
        target_df = target_df[target_df["ses_idx"] == session]
        bundle = mice_loader._build_multisubject_bundle(
            df=target_df,
            resolved_subject_ids=[subject],
            ignore_policy="include",
            features=None,
            eval_every_n=2,
            batch_size=None,
            batch_mode="single",
            metadata={"ignore_policy": "include", "eval_every_n": 2},
            trial_split_by_subject={
                subject: {"session_id": session, "total_trials": N_TRIALS, "adapt_prefix_trials": 12}
            },
        )
        bundle.metadata.update(
            {"dataset_id": "prefix-suffix", "split_strategy": "within_session_prefix_suffix"}
        )
        config = _build_finetune_config(
            yaml.safe_load((model_dir / "inputs.yaml").read_text()), model_dir
        )
        config["output"]["output_root"] = str(self.tmp / "prefix_suffix")
        config["heldout_finetuning"]["export_trial_predictions"] = True
        result = run_heldout_subject_finetuning_from_config(config, target_bundle=bundle)
        trials = pd.read_csv(result["trial_predictions_path"])
        self.assertEqual(sorted(trials.loc[trials["split"] == "eval", "trial"]), list(range(12, N_TRIALS)))
        self.assertEqual(sorted(trials.loc[trials["split"] == "train", "trial"]), list(range(12)))

    def test_trial_export_marks_only_budgeted_adaptation_sessions_as_train(self):
        model_dir = self._source_run("include")
        for budget, expected_train_sessions in ((0, 0), (1, len(HELDOUT_SUBJECTS))):
            config = _build_finetune_config(
                yaml.safe_load((model_dir / "inputs.yaml").read_text()), model_dir
            )
            config["output"]["output_root"] = str(self.tmp / f"budget_{budget}")
            config["heldout_finetuning"]["export_trial_predictions"] = True
            config["heldout_finetuning"]["adapt_sessions_per_subject"] = budget
            with mock.patch(LOAD_PATCH, return_value=(self.heldout_df.copy(), list(HELDOUT_SUBJECTS))):
                result = run_heldout_subject_finetuning_from_config(config)
            trials = pd.read_csv(result["trial_predictions_path"])
            self.assertEqual(trials.loc[trials["split"] == "train", "ses_idx"].nunique(), expected_train_sessions)
            self.assertGreater((trials["split"] == "none").sum(), 0)

    def test_sessions_without_a_role_keep_their_session_index(self):
        model_dir = self._source_run("include")
        roles = pd.read_csv(self._alternation_roles("include"), dtype=str)
        subject = HELDOUT_SUBJECTS[0]
        dropped = sorted(roles.loc[roles["subject_id"] == subject, "session_id"])[0]
        roles = roles[roles["session_id"] != dropped]
        with mock.patch(LOAD_PATCH, return_value=(self.heldout_df.copy(), list(HELDOUT_SUBJECTS))):
            bundle, report = heldout_rescore.build_rescore_bundle(
                source_run=resolve_model_run(model_dir),
                heldout_selector={"test_subject_ids": list(HELDOUT_SUBJECTS)},
                fine_tune_cfg={"batch_size": None, "batch_mode": "single"},
                roles=roles,
                swap=False,
            )
        self.assertEqual(report["sessions_in_loader_not_in_table"]["session_ids"], [dropped])
        context = {
            row["subject_id"]: row["ordered_source_session_ids"]
            for row in bundle.metadata["session_context"]["per_subject"]
        }
        self.assertEqual(len(context[subject]), N_SESSIONS)  # still packed, same indices
        self.assertEqual(bundle.metadata["session_max_index_by_subject_index"], [N_SESSIONS] * 2)
        used = bundle.metadata["train_session_ids"] + bundle.metadata["eval_session_ids"]
        self.assertNotIn(f"{subject}__{dropped}", used)
        self.assertEqual(len(used), 2 * N_SESSIONS - 1)

    def test_run_id_source_is_hydrated_into_its_own_directory(self):
        model_dir = self._source_run("include")
        out = self.tmp / "hydrated"
        with mock.patch(
            "post_training_analysis.wandb_model_dir.hydrate_model_dir", return_value=model_dir
        ) as hydrate, mock.patch(
            LOAD_PATCH, return_value=(self.heldout_df.copy(), list(HELDOUT_SUBJECTS))
        ):
            heldout_rescore.main(
                [
                    "--run-id", "run-x", "--entity", "ent", "--project", "proj",
                    "--subjects", ",".join(HELDOUT_SUBJECTS),
                    "--roles", str(self._alternation_roles("include")),
                    "--output-dir", str(out),
                ]
            )
        self.assertEqual(hydrate.call_args.kwargs["dest"], out.resolve() / "source" / "ent__proj__run-x")

    def test_session_context_plots_use_the_scoring_lambda(self):
        from model_trainers.gru_trainer import GruTrainer

        model_dir = self._source_run("include", session_conditioning=True)
        roles = self._alternation_roles("include")
        for mode, expected in (("checkpoint", 0.0), ("1", 1.0)):
            with mock.patch.object(
                GruTrainer, "_plot_subject_session_context_state_space", return_value=None
            ) as plot:
                self._rescore(model_dir, roles, "--session-lambda", mode)
            lambdas = {call.kwargs["session_curriculum_lambda"] for call in plot.call_args_list}
            self.assertEqual(lambdas, {expected})

    def test_two_way_model_has_no_ignore_probability(self):
        model_dir = self._source_run("exclude")
        out, _ = self._rescore(model_dir, self._alternation_roles("exclude"))
        trials = pd.read_csv(out / "trial_predictions.csv.gz")
        self.assertTrue(trials["p_ignore"].isna().all())
        self.assertTrue(trials["ll_3way"].isna().all())
        self.assertFalse((trials["response"] == 2).any())
        np.testing.assert_allclose(trials[["p_left", "p_right"]].sum(axis=1), 1.0, rtol=1e-5)


@unittest.skipUnless(DEPS_AVAILABLE, f"dependencies unavailable: {IMPORT_ERROR}")
class TestObservedLogLikelihoods(unittest.TestCase):
    def test_tiny_probabilities_do_not_underflow(self):
        from post_training_analysis.heldout_finetuning import _observed_log_likelihoods

        # float32 softmax rounds exp(-200) to 0; the log-softmax keeps it finite.
        logits = np.array([[0.0, -200.0, -200.0], [0.0, -200.0, 5.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        log_p, ll_observed, ll_lr = _observed_log_likelihoods(logits, np.array([1, 1, 2]))
        self.assertAlmostEqual(ll_observed[0], -200.0, places=4)
        self.assertTrue(np.isfinite(ll_observed).all())
        self.assertAlmostEqual(ll_lr[1], -200.0, places=4)  # renormalised over left/right
        self.assertTrue(np.isnan(ll_lr[2]))  # ignored trial
        np.testing.assert_allclose(np.exp(log_p).sum(axis=1), 1.0)


@unittest.skipUnless(DEPS_AVAILABLE, f"dependencies unavailable: {IMPORT_ERROR}")
class TestExplicitCheckpointStep(unittest.TestCase):
    def test_step_policy_selects_that_checkpoint_or_fails(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            case = TestHeldoutRescore()
            case.tmp = tmp
            model_dir = case._source_run("include")
            run = resolve_model_run(model_dir, checkpoint_policy="step_20")
            self.assertEqual(run.checkpoint_step, 20)
            self.assertTrue(run.params_path.endswith("step_20/params.json"))
            with self.assertRaises(FileNotFoundError):
                resolve_model_run(model_dir, checkpoint_policy="step_30")
            with self.assertRaises(ValueError):
                resolve_model_run(model_dir, checkpoint_policy="step_x")
            # An index record without params_path must not resolve to the final params.json.
            index_path = model_dir / "outputs" / "checkpoints" / "index.json"
            index = json.loads(index_path.read_text())
            index["checkpoints"].append({"step": 40, "eval_likelihood": 0.1})
            index_path.write_text(json.dumps(index))
            with self.assertRaises(FileNotFoundError):
                resolve_model_run(model_dir, checkpoint_policy="step_40")
            # Baseline RL has no checkpoint steps: an explicit step is an error, not the final fit.
            rl_dir = tmp / "baseline"
            (rl_dir / "outputs").mkdir(parents=True)
            (rl_dir / "inputs.yaml").write_text(yaml.safe_dump({"data": {}, "model": {"type": "baseline_rl"}}))
            with self.assertRaisesRegex(ValueError, "no checkpoint steps"):
                resolve_model_run(rl_dir, checkpoint_policy="step_30")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
