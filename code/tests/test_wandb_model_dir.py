"""hydrate_model_dir downloads the artifact version its caller names (wrapper #104).

external_transfer.gru_transfer checks a source artifact's digest and then hydrates it; both
must see the same version, or a version published in between would be used unchecked.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from post_training_analysis import wandb_model_dir


class TestHydrateVersion(unittest.TestCase):
    def _hydrate(self, **kwargs) -> str:
        def download(root):
            (Path(root) / "checkpoints" / "step_1").mkdir(parents=True)

        api = mock.MagicMock()
        api.artifact.return_value.download.side_effect = download
        with mock.patch.object(wandb_model_dir, "fetch_run_config", return_value={}), \
             mock.patch.object(wandb_model_dir, "_write_inputs_yaml", return_value="gru"), \
             mock.patch.object(wandb_model_dir, "_normalize_index"), \
             mock.patch.object(wandb_model_dir.wandb, "Api", return_value=api):
            wandb_model_dir.hydrate_model_dir("rid", project="p", dest=tempfile.mkdtemp(), **kwargs)
        return api.artifact.call_args.args[0]

    def test_a_named_version_is_downloaded(self):
        self.assertEqual(self._hydrate(version="v3"), "AIND-disRNN/p/gru-output-rid:v3")

    def test_the_default_stays_latest(self):
        self.assertEqual(self._hydrate(), "AIND-disRNN/p/gru-output-rid:latest")


if __name__ == "__main__":
    unittest.main()
