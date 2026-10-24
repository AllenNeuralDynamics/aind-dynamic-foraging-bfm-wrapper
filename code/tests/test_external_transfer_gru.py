"""external_transfer.gru_transfer reads its source manifest from the command line (dispatcher #179).

A job must read no study files, and study 09's freeze scripts check that each run's
``source.manifest`` / ``manifest_sha256`` match the study's manifest file. Inlining the file
text verbatim must therefore reproduce both exactly.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
