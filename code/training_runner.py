"""Shared training + held-out orchestration for the disRNN wrapper.

Both entry points delegate the train/eval body to ``run_training`` so the Code
Ocean path (run_capsule.py, config from /data/jobs) and the Beaker/HPC path
(run_hpc.py, the dispatcher's Hydra config via relative path) stay in parity:
data loading, multisubject W&B naming, held-out preload, training, held-out
evaluation, auto held-out fine-tuning (disRNN/GRU) and held-out re-fit
(baseline_rl). The caller is responsible for loading + preparing the config
(penalty multipliers, dynamic run-name components) and starting/finishing the
W&B run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydra.utils import instantiate

from base.interfaces import DatasetLoader, ModelTrainer
from utils.baseline_rl_evaluation import save_baseline_rl_output
from utils.disrnn_evaluation import (
    HeldoutEvalConfig,
    evaluate_disrnn_on_heldout_subjects,
    load_disrnn_heldout_subject_data,
)
from utils.gru_evaluation import (
    evaluate_gru_on_heldout_subjects,
    load_gru_heldout_subject_data,
)
from utils.run_helpers import resolve_heldout_test_likelihood
from post_training_analysis.heldout_finetuning import (
    run_heldout_subject_finetuning_from_config,
)

logger = logging.getLogger(__name__)


def _is_multisubject_personalized_run(hydra_config) -> bool:
    model_type = getattr(hydra_config.model, "type", None)
    architecture_cfg = getattr(hydra_config.model, "architecture", None)
    return bool(
        model_type in {"disrnn", "gru", "baseline_rl", "hb", "glmhmm"}
        and architecture_cfg is not None
        and getattr(architecture_cfg, "multisubject", False)
    )


def _auto_heldout_finetune_enabled(hydra_config) -> bool:
    """True when model.training.auto_heldout_finetune.enabled is set."""
    training_cfg = getattr(getattr(hydra_config, "model", None), "training", None)
    auto_cfg = getattr(training_cfg, "auto_heldout_finetune", None) if training_cfg else None
    return bool(getattr(auto_cfg, "enabled", False)) if auto_cfg is not None else False


def _baseline_rl_heldout_refit_enabled(hydra_config) -> bool:
    """True when model.heldout_refit.enabled is set (baseline_rl held-out re-fit)."""
    model_cfg = getattr(hydra_config, "model", None)
    refit_cfg = getattr(model_cfg, "heldout_refit", None) if model_cfg is not None else None
    return bool(getattr(refit_cfg, "enabled", False)) if refit_cfg is not None else False


def _baseline_rl_skip_train_fit(hydra_config) -> bool:
    """True when baseline_rl should run only the held-out re-fit path."""
    model_cfg = getattr(hydra_config, "model", None)
    refit_cfg = getattr(model_cfg, "heldout_refit", None) if model_cfg is not None else None
    return bool(getattr(refit_cfg, "skip_train_fit", False)) if refit_cfg is not None else False


def _baseline_rl_heldout_loader_kwargs(data_cfg) -> dict:
    """Build held-out loader kwargs, honoring explicit held-out subject subsets.

    ``subject_ids`` is always set. Explicit ids bypass ``split="heldout"`` in the loader, so
    inheriting a training-cohort ``data.subject_ids`` would score seen subjects as held-out;
    ``None`` restores the reserved held-out cohort unless ``test_subject_ids`` picks a subset.
    """
    kwargs = {"split": "heldout", "multisubject": True, "subject_ids": None}
    test_subject_ids = getattr(data_cfg, "test_subject_ids", None)
    if test_subject_ids is not None:
        kwargs["subject_ids"] = list(test_subject_ids)
    return kwargs


def _run_auto_heldout_finetune(hydra_config, *, model_type, wandb_run, model_dir="/results"):
    """Auto-run held-out subject fine-tuning + eval after multisubject training.

    Synthesizes an in-memory fine-tuning config pointing at the just-finished run
    (``model_dir``, which must contain ``inputs.yaml`` and ``outputs/``) and logs
    aggregate held-out train/eval likelihood into the main W&B run under a
    ``heldout/`` namespace. Returns the fine-tuning summary dict.
    """
    auto_cfg = getattr(hydra_config.model.training, "auto_heldout_finetune", None)
    finetune_config = {
        "source_run": {
            "model_dir": str(model_dir),
            "checkpoint_policy": str(getattr(auto_cfg, "checkpoint_policy", "final")),
        },
        # All None -> inherit the reserved held-out set from <model_dir>/inputs.yaml.
        "heldout_subjects": {
            key: None
            for key in (
                "test_subject_ids",
                "curricula",
                "min_sessions",
                "heldout_every_n",
                "mature_only",
                "cols_to_retain",
            )
        },
        "heldout_finetuning": {
            "n_steps": int(getattr(auto_cfg, "n_steps", 50)),
            "lr": float(getattr(auto_cfg, "lr", 1e-2)),
            "checkpoint_every_n_steps": int(
                getattr(auto_cfg, "checkpoint_every_n_steps", 10)
            ),
            "batch_size": getattr(auto_cfg, "batch_size", None),
            "batch_mode": str(getattr(auto_cfg, "batch_mode", "single")),
            "keep_media_files": bool(getattr(auto_cfg, "keep_media_files", True)),
            "checkpoint_plot_split_examples_every_n": int(
                getattr(auto_cfg, "checkpoint_plot_split_examples_every_n", 10)
            ),
            "checkpoint_save_output_df_every_n": int(
                getattr(auto_cfg, "checkpoint_save_output_df_every_n", 0)
            ),
            "train_example_sessions_per_subject": int(
                getattr(auto_cfg, "train_example_sessions_per_subject", 1)
            ),
            "eval_example_sessions_per_subject": int(
                getattr(auto_cfg, "eval_example_sessions_per_subject", 1)
            ),
            "example_max_subjects": int(getattr(auto_cfg, "example_max_subjects", 6)),
        },
        "output": {
            "output_root": str(
                getattr(auto_cfg, "output_root", "/results/heldout_subject_finetuning")
            ),
            "run_name_suffix": getattr(auto_cfg, "run_name_suffix", None),
        },
        "seed": getattr(hydra_config, "seed", None),
        # No "wandb" key: held-out metrics are logged into the injected main run.
    }
    wandb_step_offset = (
        int(getattr(wandb_run, "step", 0)) + 1 if wandb_run is not None else None
    )
    logger.info(
        "Auto held-out fine-tuning for multisubject %s: n_steps=%s lr=%s checkpoint_policy=%s",
        str(model_type).upper(),
        finetune_config["heldout_finetuning"]["n_steps"],
        finetune_config["heldout_finetuning"]["lr"],
        finetune_config["source_run"]["checkpoint_policy"],
    )
    return run_heldout_subject_finetuning_from_config(
        finetune_config,
        wandb_run=wandb_run,
        wandb_key_prefix="heldout",
        wandb_step_offset=wandb_step_offset,
    )


def run_training(hydra_config, wandb_run=None, run_output_dir="/results"):
    """Load data, train the model, and run held-out evaluation/fine-tuning.

    Assumes the config is already prepared and the W&B run already started.
    Returns the trainer's output payload.
    """
    data_cfg = hydra_config.data
    model_type = getattr(hydra_config.model, "type", None)
    heldout_cfg = HeldoutEvalConfig.from_data_cfg(
        data_cfg,
        default_example_max_subjects=int(getattr(hydra_config, "example_max_subjects", 6)),
    )

    if (
        model_type == "baseline_rl"
        and hasattr(data_cfg, "mature_only")
        and heldout_cfg.enabled
        and _baseline_rl_heldout_refit_enabled(hydra_config)
        and _baseline_rl_skip_train_fit(hydra_config)
    ):
        if hasattr(data_cfg, "mature_only"):
            logger.info(
                "Snapshot data loading: mature_only=%s, subject_ids=%s, start=%s, end=%s",
                data_cfg.mature_only,
                getattr(data_cfg, "subject_ids", None),
                getattr(data_cfg, "subject_start", None),
                getattr(data_cfg, "subject_end", None),
            )

        model_trainer: ModelTrainer = instantiate(hydra_config.model)
        loggers = {"wandb": wandb_run} if wandb_run is not None else None

        logger.info(
            "Skipping baseline RL training-subject fit; fitting reserved held-out "
            "subjects directly under heldout/*."
        )
        heldout_loader_kwargs = _baseline_rl_heldout_loader_kwargs(data_cfg)
        if heldout_loader_kwargs.get("subject_ids") is not None:
            logger.info(
                "Baseline RL held-out subject override: %s",
                heldout_loader_kwargs["subject_ids"],
            )
        heldout_loader = instantiate(hydra_config.data, **heldout_loader_kwargs)
        heldout_bundle = heldout_loader.load()
        logger.info("Loaded held-out dataset bundle with metadata: %s", heldout_bundle.metadata)

        if wandb_run is not None and bool(heldout_bundle.metadata.get("multisubject", False)):
            current_name = wandb_run.name or ""
            if "multisubject" not in current_name.lower():
                new_name = f"{current_name}_multisubject" if current_name else "multisubject"
                wandb_run.name = new_name
                logger.info("Updated wandb run name to reflect multisubject mode: %s", new_name)
            wandb_run.config.update({"multisubject": True})

        resolved_subject_ids = heldout_bundle.metadata.get("subject_ids")
        if wandb_run is not None and resolved_subject_ids is not None:
            n_subj = len(resolved_subject_ids)
            current_name = wandb_run.name or ""
            suffix = f"n{n_subj}subj"
            if suffix not in current_name:
                new_name = f"{current_name}_{suffix}" if current_name else suffix
                wandb_run.name = new_name
                logger.info("Updated wandb run name to: %s", new_name)
            wandb_run.config.update({"resolved_subject_ids": list(resolved_subject_ids)})

        heldout_summary = model_trainer.fit_heldout(heldout_bundle, loggers=loggers)
        heldout_test_likelihood = resolve_heldout_test_likelihood(heldout_summary)
        if heldout_test_likelihood is not None:
            logger.info(
                "Held-out re-fit eval likelihood: %.4f", float(heldout_test_likelihood)
            )

        output = {
            "type": "baseline_rl",
            "train_fit_skipped": True,
            "fit_strategy": "heldout_refit_only",
            "heldout_test": heldout_summary,
        }
        output_path = save_baseline_rl_output(
            getattr(hydra_config.model, "output_dir", "/results/outputs"),
            output,
            indent=4,
        )
        logger.info(
            "Saved baseline RL held-out-only output with held-out re-fit summary at %s",
            output_path,
        )
        return output

    # --- Load data ---
    dataset_loader: DatasetLoader = instantiate(hydra_config.data)

    # When using snapshot-based loading, surface the key selection parameters
    # so they are clearly visible at the top of the run log.
    if hasattr(data_cfg, "mature_only"):
        logger.info(
            "Snapshot data loading: mature_only=%s, subject_ids=%s, start=%s, end=%s",
            data_cfg.mature_only,
            getattr(data_cfg, "subject_ids", None),
            getattr(data_cfg, "subject_start", None),
            getattr(data_cfg, "subject_end", None),
        )

    dataset_bundle = dataset_loader.load()
    logger.info("Loaded dataset bundle with metadata: %s", dataset_bundle.metadata)
    is_multisubject_dataset = bool(dataset_bundle.metadata.get("multisubject", False))

    if wandb_run is not None and is_multisubject_dataset:
        current_name = wandb_run.name or ""
        if "multisubject" not in current_name.lower():
            new_name = f"{current_name}_multisubject" if current_name else "multisubject"
            wandb_run.name = new_name
            logger.info("Updated wandb run name to reflect multisubject mode: %s", new_name)
        wandb_run.config.update({"multisubject": True})

    # Append the resolved subject COUNT to the wandb run name so rank-based slices
    # (where subject_ids is null in config) stay identifiable in the UI without a
    # multi-hundred-ID string. The full list lives in config.resolved_subject_ids.
    resolved_subject_ids = dataset_bundle.metadata.get("subject_ids")
    if wandb_run is not None and resolved_subject_ids is not None:
        n_subj = len(resolved_subject_ids)
        current_name = wandb_run.name or ""
        suffix = f"n{n_subj}subj"
        if suffix not in current_name:  # idempotent across autoResume restarts
            new_name = f"{current_name}_{suffix}" if current_name else suffix
            wandb_run.name = new_name
            logger.info("Updated wandb run name to: %s", new_name)
        wandb_run.config.update({"resolved_subject_ids": list(resolved_subject_ids)})

    heldout_test_data = None
    is_multisubject_personalized_model = _is_multisubject_personalized_run(hydra_config)
    if (
        model_type in {"disrnn", "gru"}
        and hasattr(hydra_config.data, "mature_only")
        and heldout_cfg.enabled
        and not is_multisubject_personalized_model
    ):
        try:
            if model_type == "disrnn":
                heldout_test_data = load_disrnn_heldout_subject_data(heldout_cfg)
            elif model_type == "gru":
                heldout_test_data = load_gru_heldout_subject_data(heldout_cfg)
        except Exception as exc:
            logger.warning(
                "Preloading held-out test data failed; evaluation will fall back to lazy loading: %s",
                exc,
            )
    elif is_multisubject_personalized_model and heldout_cfg.enabled:
        logger.info(
            "Skipping held-out test-data PRELOAD (only used by the per-checkpoint "
            "held-out eval) for multisubject %s; v1 supports seen-subject personalization "
            "only. The end-of-training held-out fine-tune (auto_heldout_finetune) loads "
            "its own held-out set from inputs.yaml and is unaffected.",
            str(model_type).upper(),
        )

    # The hierarchical Bayesian and GLM-HMM trainers score held-out subjects themselves, and
    # need their trials to come from the same loader as the training split -- otherwise the
    # two are filtered independently and the comparison against the neural models is void.
    # The GLM-HMM also takes the held-out loader's train/eval session split from its metadata,
    # so it conditions on and scores exactly the sessions the GRU fine-tune does.
    if model_type in {"hb", "glmhmm"} and heldout_cfg.enabled:
        heldout_loader = instantiate(
            hydra_config.data, **_baseline_rl_heldout_loader_kwargs(data_cfg)
        )
        hb_heldout_bundle = heldout_loader.load()
        dataset_bundle.extras["heldout_raw"] = hb_heldout_bundle.raw
        dataset_bundle.extras["heldout_metadata"] = hb_heldout_bundle.metadata
        logger.info(
            "%s held-out cohort loaded from the same loader: %s",
            str(model_type).upper(),
            hb_heldout_bundle.metadata,
        )

    # --- Train model ---
    model_trainer: ModelTrainer = instantiate(
        hydra_config.model,
        heldout_data=heldout_test_data,
    )
    loggers = {"wandb": wandb_run} if wandb_run is not None else None
    output = model_trainer.fit(dataset_bundle, loggers=loggers)
    if heldout_test_data is None:
        heldout_test_data = getattr(model_trainer, "heldout_data", None)

    checkpoint_heldout_summaries = []
    if isinstance(output, dict):
        raw_checkpoint_heldout = output.get("heldout_test_checkpoints", [])
        if isinstance(raw_checkpoint_heldout, list):
            checkpoint_heldout_summaries = [
                item for item in raw_checkpoint_heldout if isinstance(item, dict)
            ]

    if (
        model_type == "baseline_rl"
        and hasattr(hydra_config.data, "mature_only")
        and heldout_cfg.enabled
        and _baseline_rl_heldout_refit_enabled(hydra_config)
    ):
        # Fit a fresh RL agent on each reserved held-out subject (train/eval split)
        # and log aggregate held-out train/eval likelihood under heldout/*, like the
        # NN models. The held-out reserved set is inherently multiple subjects, so
        # load it in multisubject mode regardless of the training run's mode.
        heldout_summary = None
        try:
            heldout_loader_kwargs = _baseline_rl_heldout_loader_kwargs(data_cfg)
            if heldout_loader_kwargs.get("subject_ids") is not None:
                logger.info(
                    "Baseline RL held-out subject override: %s",
                    heldout_loader_kwargs["subject_ids"],
                )
            heldout_loader = instantiate(hydra_config.data, **heldout_loader_kwargs)
            heldout_bundle = heldout_loader.load()
            heldout_summary = model_trainer.fit_heldout(heldout_bundle, loggers=loggers)
        except Exception as exc:
            logger.warning("Held-out re-fit failed and will be skipped: %s", exc)
            heldout_summary = {
                "enabled": True,
                "heldout_refit_failed": True,
                "error": str(exc),
            }
        heldout_test_likelihood = resolve_heldout_test_likelihood(heldout_summary)
        if heldout_test_likelihood is not None:
            logger.info(
                "Held-out re-fit eval likelihood: %.4f", float(heldout_test_likelihood)
            )
        if isinstance(output, dict):
            output["heldout_test"] = heldout_summary
        output_path = save_baseline_rl_output(
            getattr(hydra_config.model, "output_dir", "/results/outputs"),
            output,
            indent=4,
        )
        logger.info("Updated baseline RL output with held-out re-fit summary at %s", output_path)
    elif (
        hasattr(hydra_config.data, "mature_only")
        and heldout_cfg.enabled
        and not is_multisubject_personalized_model
    ):
        heldout_summary = None
        try:
            if model_type == "disrnn":
                training_cfg = getattr(hydra_config.model, "training", {})
                checkpoint_keep_media_files = bool(
                    getattr(training_cfg, "checkpoint_keep_media_files", True)
                )
                total_steps = int(
                    getattr(getattr(hydra_config.model, "training", {}), "n_steps", 0)
                )
                evaluated_checkpoint_steps = {
                    int(item.get("step", -1))
                    for item in checkpoint_heldout_summaries
                    if isinstance(item, dict)
                }
                if total_steps in evaluated_checkpoint_steps:
                    logger.info(
                        "Skipping duplicate final held-out eval; step_%s already evaluated in checkpoint loop.",
                        total_steps,
                    )
                    heldout_summary = next(
                        (
                            item
                            for item in checkpoint_heldout_summaries
                            if int(item.get("step", -1)) == total_steps
                        ),
                        None,
                    )
                else:
                    heldout_summary = evaluate_disrnn_on_heldout_subjects(
                        hydra_config,
                        wandb_run=wandb_run,
                        output_subdir=f"heldout_test/checkpoints/step_{total_steps}",
                        log_to_wandb=False,
                        heldout_data=heldout_test_data,
                        log_scope="Final",
                    )
                    if heldout_summary is not None and wandb_run is not None:
                        warmup_steps = int(getattr(training_cfg, "n_warmup_steps", 0))
                        wandb_step = warmup_steps + total_steps
                        wandb_run.log(
                            {
                                "checkpoint/heldout_test_likelihood": float(
                                    heldout_summary["test_likelihood"]
                                ),
                                "checkpoint/step": total_steps,
                            },
                            step=wandb_step,
                        )
                        try:
                            import wandb

                            trial_plot_paths = heldout_summary.get("plots", {}).get(
                                "latents_over_trials_examples", []
                            )
                            space_plot_paths = heldout_summary.get("plots", {}).get(
                                "latents_in_space_examples", []
                            )
                            checkpoint_plot_payload = {}
                            if trial_plot_paths:
                                checkpoint_plot_payload[
                                    "checkpoint/heldout/latents_over_trials_examples"
                                ] = [wandb.Image(str(path)) for path in trial_plot_paths]
                            if space_plot_paths:
                                checkpoint_plot_payload[
                                    "checkpoint/heldout/latents_in_space_examples"
                                ] = [wandb.Image(str(path)) for path in space_plot_paths]
                            if checkpoint_plot_payload:
                                wandb_run.log(checkpoint_plot_payload, step=wandb_step)
                                if not checkpoint_keep_media_files:
                                    for path in trial_plot_paths + space_plot_paths:
                                        try:
                                            Path(str(path)).unlink(missing_ok=True)
                                        except Exception as path_exc:
                                            logger.warning(
                                                "Failed to remove final held-out media file %s: %s",
                                                path,
                                                path_exc,
                                            )
                                    plots = heldout_summary.get("plots", {})
                                    if isinstance(plots, dict):
                                        plots["latents_over_trials_examples"] = []
                                        plots["latents_in_space_examples"] = []
                        except Exception as exc:
                            logger.warning(
                                "Final held-out image logging failed for step=%s: %s",
                                total_steps,
                                exc,
                            )
            elif model_type == "gru":
                training_cfg = getattr(hydra_config.model, "training", {})
                checkpoint_keep_media_files = bool(
                    getattr(training_cfg, "checkpoint_keep_media_files", True)
                )
                total_steps = int(
                    getattr(getattr(hydra_config.model, "training", {}), "n_steps", 0)
                )
                evaluated_checkpoint_steps = {
                    int(item.get("step", -1))
                    for item in checkpoint_heldout_summaries
                    if isinstance(item, dict)
                }
                if total_steps in evaluated_checkpoint_steps:
                    logger.info(
                        "Skipping duplicate final held-out eval; step_%s already evaluated in checkpoint loop.",
                        total_steps,
                    )
                    heldout_summary = next(
                        (
                            item
                            for item in checkpoint_heldout_summaries
                            if int(item.get("step", -1)) == total_steps
                        ),
                        None,
                    )
                else:
                    heldout_summary = evaluate_gru_on_heldout_subjects(
                        hydra_config,
                        wandb_run=wandb_run,
                        output_subdir=f"heldout_test/checkpoints/step_{total_steps}",
                        log_to_wandb=False,
                        heldout_data=heldout_test_data,
                        log_scope="Final",
                    )
                    if heldout_summary is not None and wandb_run is not None:
                        wandb_step = total_steps
                        wandb_run.log(
                            {
                                "checkpoint/heldout_test_likelihood": float(
                                    heldout_summary["test_likelihood"]
                                ),
                                "checkpoint/step": total_steps,
                            },
                            step=wandb_step,
                        )
                        try:
                            import wandb

                            trial_plot_paths = heldout_summary.get("plots", {}).get(
                                "latents_over_trials_examples", []
                            )
                            space_plot_paths = heldout_summary.get("plots", {}).get(
                                "latents_in_space_examples", []
                            )
                            checkpoint_plot_payload = {}
                            if trial_plot_paths:
                                checkpoint_plot_payload[
                                    "checkpoint/heldout/latents_over_trials_examples"
                                ] = [wandb.Image(str(path)) for path in trial_plot_paths]
                            if space_plot_paths:
                                checkpoint_plot_payload[
                                    "checkpoint/heldout/latents_in_space_examples"
                                ] = [wandb.Image(str(path)) for path in space_plot_paths]
                            if checkpoint_plot_payload:
                                wandb_run.log(checkpoint_plot_payload, step=wandb_step)
                                if not checkpoint_keep_media_files:
                                    for path in trial_plot_paths + space_plot_paths:
                                        try:
                                            Path(str(path)).unlink(missing_ok=True)
                                        except Exception as path_exc:
                                            logger.warning(
                                                "Failed to remove final held-out media file %s: %s",
                                                path,
                                                path_exc,
                                            )
                                    plots = heldout_summary.get("plots", {})
                                    if isinstance(plots, dict):
                                        plots["latents_over_trials_examples"] = []
                                        plots["latents_in_space_examples"] = []
                        except Exception as exc:
                            logger.warning(
                                "Final held-out image logging failed for step=%s: %s",
                                total_steps,
                                exc,
                            )
        except Exception as exc:
            logger.warning("Held-out evaluation failed and will be skipped: %s", exc)
            heldout_summary = {
                "enabled": True,
                "evaluation_failed": True,
                "error": str(exc),
            }

        if heldout_summary is not None:
            heldout_test_likelihood = resolve_heldout_test_likelihood(heldout_summary)
            if heldout_test_likelihood is not None:
                logger.info(
                    "Final held-out test likelihood: %.4f",
                    float(heldout_test_likelihood),
                )
            if isinstance(output, dict):
                output["heldout_test"] = heldout_summary
    elif is_multisubject_personalized_model and heldout_cfg.enabled:
        if model_type in {"disrnn", "gru"} and _auto_heldout_finetune_enabled(hydra_config):
            try:
                auto_summary = _run_auto_heldout_finetune(
                    hydra_config,
                    model_type=model_type,
                    wandb_run=wandb_run,
                    model_dir=run_output_dir,
                )
            except Exception as exc:
                logger.warning(
                    "Auto held-out fine-tuning failed and will be skipped: %s", exc
                )
                auto_summary = {
                    "enabled": True,
                    "auto_finetune_failed": True,
                    "error": str(exc),
                }
            if isinstance(output, dict):
                output["heldout_finetune"] = auto_summary
        elif model_type in {"disrnn", "gru"}:
            # hb and glmhmm score held-out subjects inside fit(); this note is only true of
            # the neural models.
            logger.info(
                "Skipping end-of-training held-out fine-tune for multisubject %s: "
                "auto_heldout_finetune.enabled is FALSE (this run logs seen-subject "
                "metrics only, NO heldout/* metrics). Set "
                "model.training.auto_heldout_finetune.enabled=true to get held-out-mouse "
                "transfer likelihood.",
                str(model_type).upper(),
            )

    return output
