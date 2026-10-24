"""Unit tests for run-name helper utilities."""

from __future__ import annotations

import os
import unittest
from unittest import mock

try:
    from omegaconf import OmegaConf

    from utils.run_helpers import (
        _code_versions,
        _launch_meta,
        apply_dynamic_run_name_components,
        apply_model_penalty_multipliers,
        resolve_disrnn_penalties,
        resolve_heldout_test_likelihood,
    )

    RUN_HELPERS_DEPS_AVAILABLE = True
    RUN_HELPERS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    RUN_HELPERS_DEPS_AVAILABLE = False
    RUN_HELPERS_IMPORT_ERROR = exc


@unittest.skipUnless(
    RUN_HELPERS_DEPS_AVAILABLE,
    f"run helper dependencies unavailable: {RUN_HELPERS_IMPORT_ERROR}",
)
class TestRunHelpers(unittest.TestCase):
    def test_code_versions_prefers_all_runtime_commit_env_vars(self):
        commits = {
            "WRAPPER_COMMIT": "wrapper-sha",
            "DISPATCHER_COMMIT": "dispatcher-sha",
            "FORAGING_MODELS_COMMIT": "foraging-models-sha",
        }

        with mock.patch.dict(os.environ, commits, clear=False):
            versions = _code_versions()

        self.assertEqual(versions["wrapper_commit"], "wrapper-sha")
        self.assertEqual(versions["dispatcher_commit"], "dispatcher-sha")
        self.assertEqual(
            versions["foraging_models_commit"], "foraging-models-sha"
        )

    _META_ENV_KEYS = (
        "BFM_META_STUDY", "BFM_META_VARIANT", "BFM_META_LAUNCH_ID", "BFM_META_LABEL",
        "BFM_META_NOTE", "BFM_META_CONFIG_HASH", "BFM_META_STUDIES_COMMIT",
    )

    def _env_without_meta(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k not in self._META_ENV_KEYS}

    def test_launch_meta_records_the_studies_commit_from_env(self):
        env = {
            **self._env_without_meta(),
            "BFM_META_STUDY": "08-hb-vs-gru-heldout",
            "BFM_META_STUDIES_COMMIT": "s" * 40,
        }

        with mock.patch.dict(os.environ, env, clear=True):
            meta = _launch_meta({"launch_id": "20261023-120000"})

        self.assertEqual(meta["studies_commit"], "s" * 40)
        self.assertEqual(meta["study"], "08-hb-vs-gru-heldout")
        # config-side values the env did not set are kept
        self.assertEqual(meta["launch_id"], "20261023-120000")

    def test_launch_meta_without_the_env_keeps_the_config_meta_unchanged(self):
        config_meta = {"study": "01-gru-scaling-law", "studies_commit": "c" * 40}

        with mock.patch.dict(os.environ, self._env_without_meta(), clear=True):
            meta = _launch_meta(config_meta)

        # a Hydra-injected `+meta.studies_commit` (the HPC route) survives an unset env
        self.assertEqual(meta, config_meta)

    def test_launch_meta_env_wins_over_the_config_value(self):
        env = {**self._env_without_meta(), "BFM_META_STUDIES_COMMIT": "e" * 40}

        with mock.patch.dict(os.environ, env, clear=True):
            meta = _launch_meta({"studies_commit": "c" * 40})

        self.assertEqual(meta["studies_commit"], "e" * 40)

    def test_resolve_disrnn_penalties_uses_beta_as_default_base(self):
        resolved = resolve_disrnn_penalties(
            {
                "beta": 1e-3,
                "latent_penalty": 1e-3,
                "update_net_latent_penalty_multiplier": 10,
            }
        )

        self.assertEqual(resolved["update_net_latent_penalty"], 1e-2)
        self.assertEqual(resolved["latent_penalty"], 1e-3)
        self.assertNotIn("update_net_latent_penalty_multiplier", resolved)

    def test_apply_model_penalty_multipliers_updates_disrnn_config(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "type": "disrnn",
                    "penalties": {
                        "beta": 1e-3,
                        "update_net_latent_penalty": "${.beta}",
                        "update_net_latent_penalty_multiplier": 10,
                    },
                }
            }
        )

        apply_model_penalty_multipliers(cfg)

        self.assertEqual(cfg.model.penalties.update_net_latent_penalty, 1e-2)
        self.assertFalse("update_net_latent_penalty_multiplier" in cfg.model.penalties)

    def test_apply_dynamic_run_name_components_appends_multisubject_once(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "multisubject": True,
                    "run_name_component": "mice_snapshot",
                },
                "model": {
                    "run_name_component": "disrnn_beta0.001_lr0.001",
                    "architecture": {"multisubject": True},
                },
            }
        )

        apply_dynamic_run_name_components(cfg)

        self.assertEqual(cfg.data.run_name_component, "mice_snapshot_multisubject")
        self.assertEqual(
            cfg.model.run_name_component,
            "disrnn_beta0.001_lr0.001_multisubject",
        )

        apply_dynamic_run_name_components(cfg)

        self.assertEqual(cfg.data.run_name_component, "mice_snapshot_multisubject")
        self.assertEqual(
            cfg.model.run_name_component,
            "disrnn_beta0.001_lr0.001_multisubject",
        )


    # --- Phase A regression guards (bugs #3/#4) -------------------------------

    def test_resolve_heldout_test_likelihood_reads_test_likelihood(self):
        # Fresh-eval summaries use "test_likelihood".
        self.assertEqual(
            resolve_heldout_test_likelihood({"test_likelihood": 0.5}), 0.5
        )

    def test_resolve_heldout_test_likelihood_reads_dedup_key(self):
        # The dedup-hit path reuses a checkpoint summary keyed differently.
        self.assertEqual(
            resolve_heldout_test_likelihood({"heldout_test_likelihood": 0.7}), 0.7
        )

    def test_resolve_heldout_test_likelihood_handles_missing_and_non_dict(self):
        # Bug #3/#4: failure-fallback summaries and non-dicts must not blow up.
        self.assertIsNone(
            resolve_heldout_test_likelihood({"enabled": True, "evaluation_failed": True})
        )
        self.assertIsNone(resolve_heldout_test_likelihood(None))
        self.assertIsNone(resolve_heldout_test_likelihood("not a dict"))

    def test_resolve_heldout_test_likelihood_zero_is_preserved(self):
        # 0.0 is a valid likelihood and must not be treated as "missing".
        self.assertEqual(
            resolve_heldout_test_likelihood({"test_likelihood": 0.0}), 0.0
        )


if __name__ == "__main__":
    unittest.main()
