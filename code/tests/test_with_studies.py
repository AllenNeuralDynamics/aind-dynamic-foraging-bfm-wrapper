"""beaker/with_studies.sh fetches the studies repo at the pinned commit, then runs the job.

Exercised against a throwaway local repository (``STUDIES_REPO_URL=file://...``) so the
test needs no network and never touches the real studies repository.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "beaker" / "with_studies.sh"


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


class TestWithStudies(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="with_studies_"))
        self.origin = self.tmp / "origin"
        (self.origin / "studies" / "09-x").mkdir(parents=True)
        (self.origin / "studies" / "09-x" / "run.py").write_text("print('ran')\n")
        _git(self.tmp, "init", "-q", str(self.origin))
        _git(self.origin, "config", "user.email", "t@example.com")
        _git(self.origin, "config", "user.name", "t")
        _git(self.origin, "config", "uploadpack.allowReachableSHA1InWant", "true")
        _git(self.origin, "add", "-A")
        _git(self.origin, "commit", "-q", "-m", "first")
        self.first = _git(self.origin, "rev-parse", "HEAD")
        (self.origin / "studies" / "09-x" / "run.py").write_text("print('second')\n")
        _git(self.origin, "commit", "-q", "-am", "second")

    def _run(self, *cmd: str, **env: str) -> subprocess.CompletedProcess:
        base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        base.update({"STUDIES_ROOT": str(self.tmp / "checkout"),
                     "STUDIES_REPO_URL": f"file://{self.origin}", **env})
        return subprocess.run(["bash", str(SCRIPT), *cmd], capture_output=True, text=True, env=base)

    def test_checks_out_the_pinned_commit_and_exports_it(self):
        out = self._run("bash", "-c", 'echo "$STUDIES_COMMIT"; cat "$STUDIES_ROOT/studies/09-x/run.py"',
                        STUDIES_REF=self.first)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.strip().splitlines()
        # the pinned commit, not the newer tip
        self.assertEqual(lines[-2], self.first)
        self.assertEqual(lines[-1], "print('ran')")

    def test_a_second_run_reuses_the_checkout(self):
        self.assertEqual(self._run("true", STUDIES_REF=self.first).returncode, 0)
        out = self._run("bash", "-c", 'echo "$STUDIES_COMMIT"', STUDIES_REF=self.first)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], self.first)

    def test_requires_a_ref(self):
        out = self._run("true")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("STUDIES_REF is required", out.stderr)

    def test_requires_a_command(self):
        out = self._run(STUDIES_REF=self.first)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("no command given", out.stderr)


if __name__ == "__main__":
    unittest.main()
