"""external_transfer.gru_transfer reads its source manifest from the command line (dispatcher #179).

A job must read no study files, and study 09's freeze scripts check that each run's
``source.manifest`` / ``manifest_sha256`` match the study's manifest file. Inlining the file
text verbatim must therefore reproduce both exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from external_transfer import gru_transfer

MANIFEST = json.dumps({
    "entity": "AIND-disRNN", "project": "mice_data_scaling", "group": "g",
    "checkpoint_policy": "best_eval", "embedding_finetuning": {"n_steps": 500, "lr": 0.001},
    "runs": {"d10-s0": {"run_id": "abc", "artifact_digest": "d" * 32}},
}, indent=2) + "\n"


class TestSourceManifest(unittest.TestCase):
    def _args(self, *argv: str):
        return gru_transfer._parser().parse_args(["--dataset", "chen", "--source-key", "d10-s0", *argv])

    def test_inline_text_is_hashed_as_given_and_keeps_the_study_file_name(self):
        args = self._args("--source-manifest-json", MANIFEST, "--source-manifest-name", "source_runs_e8.json")
        raw, name, manifest = gru_transfer._load_source_manifest(args)
        self.assertEqual(name, "source_runs_e8.json")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), hashlib.sha256(MANIFEST.encode()).hexdigest())
        self.assertEqual(manifest["runs"]["d10-s0"]["run_id"], "abc")

    def test_a_file_gives_the_same_digest_as_its_inlined_text(self):
        path = Path(tempfile.mkdtemp()) / "source_runs.json"
        path.write_text(MANIFEST)
        from_file = gru_transfer._load_source_manifest(self._args("--source-manifest", str(path)))
        inline = gru_transfer._load_source_manifest(self._args("--source-manifest-json", path.read_text()))
        self.assertEqual(from_file[0], inline[0])
        self.assertEqual(from_file[1], inline[1])   # both "source_runs.json"

    def test_a_manifest_source_is_required(self):
        with self.assertRaises(SystemExit):
            gru_transfer._parser().parse_args(["--dataset", "chen", "--source-key", "d10-s0"])



class TestAdaptationBudget(unittest.TestCase):
    """dispatcher #166: K-shot curves reuse the runner with a per-subject session budget."""

    def _args(self, *argv: str):
        return gru_transfer._parser().parse_args(
            ["--dataset", "chen", "--source-key", "d614-s0", "--source-manifest-json", MANIFEST, *argv])

    def test_omitting_the_budget_keeps_the_matched_half_condition(self):
        args = self._args()
        self.assertIsNone(args.adapt_sessions_per_subject)
        self.assertEqual(gru_transfer._condition(args.adapt_sessions_per_subject), "matched_half")

    def test_a_budget_names_its_condition_and_zero_is_zero_shot(self):
        self.assertEqual(self._args("--adapt-sessions-per-subject", "4").adapt_sessions_per_subject, 4)
        self.assertEqual(gru_transfer._condition(0), "k0")

    def test_a_negative_budget_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._args("--adapt-sessions-per-subject", "-1")


class TestRunMeta(unittest.TestCase):
    def test_the_studies_commit_from_the_launcher_reaches_the_run_config(self):
        env = {"BFM_META_STUDIES_COMMIT": "s" * 40, "DISPATCHER_COMMIT": "d" * 40,
               "BFM_META_STUDY": "09-gru-cross-species-transfer"}
        with mock.patch.dict(os.environ, env, clear=False):
            meta = gru_transfer._run_meta()
        self.assertEqual(meta["studies_commit"], "s" * 40)
        self.assertEqual(meta["dispatcher_commit"], "d" * 40)
        self.assertEqual(meta["study"], "09-gru-cross-species-transfer")

if __name__ == "__main__":
    unittest.main()
