"""install.sh: fetches the Studio into a directory from an archive of its own
repository. Driven with STUDIO_ARCHIVE, so no network is needed."""

import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[2] / "install.sh"


def make_archive(root: Path, name: str, files: dict) -> Path:
    """A .tar.gz with one top-level directory, as GitHub serves a repository."""
    top = root / "src" / name
    for rel, text in files.items():
        path = top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    archive = root / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(top, arcname=name)
    return archive


class InstallScriptTests(unittest.TestCase):
    STUDIO = {"run.sh": "#!/bin/sh\n", "setup.sh": "#!/bin/sh\necho setup-ran > .setup-ran\n",
              "README.md": "studio\n", "src/python/ui/main.py": "print('ui')\n"}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.archive = make_archive(self.tmp, "neat-genai-studio-main", self.STUDIO)
        self.work = self.tmp / "work"
        self.work.mkdir()

    def install(self, *args, archive=None, **env):
        full = dict(os.environ, STUDIO_ARCHIVE=str(archive or self.archive), **env)
        full.pop("GITHUB_TOKEN", None)
        return subprocess.run(["bash", str(INSTALL_SH), *args], cwd=self.work, env=full,
                              capture_output=True, text=True, timeout=60)

    def test_installs_into_the_default_directory(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.work / "neat-genai-studio"
        self.assertTrue((target / "src/python/ui/main.py").is_file())   # top level stripped
        self.assertTrue(os.access(target / "run.sh", os.X_OK))
        self.assertTrue(os.access(target / "setup.sh", os.X_OK))
        self.assertIn("./setup.sh", result.stdout)
        self.assertFalse((target / ".setup-ran").exists())              # setup is not run unasked

    def test_directory_from_the_argument_or_the_environment(self):
        self.assertEqual(self.install("by-arg").returncode, 0)
        self.assertTrue((self.work / "by-arg/run.sh").is_file())
        self.assertEqual(self.install(STUDIO_DIR="by-env").returncode, 0)
        self.assertTrue((self.work / "by-env/run.sh").is_file())

    def test_setup_runs_only_when_asked(self):
        result = self.install(STUDIO_SETUP="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.work / "neat-genai-studio/.setup-ran").is_file())

    def test_an_existing_install_is_left_alone(self):
        self.install()
        marker = self.work / "neat-genai-studio/config.local.yaml"
        marker.write_text("mine\n")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("./run.sh update", result.stderr)
        self.assertEqual(marker.read_text(), "mine\n")

    def test_a_non_empty_unrelated_directory_is_refused(self):
        (self.work / "busy").mkdir()
        (self.work / "busy/notes.txt").write_text("keep\n")
        result = self.install("busy")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sorted(p.name for p in (self.work / "busy").iterdir()), ["notes.txt"])

    def test_an_archive_that_is_not_the_studio_is_refused_and_leaves_nothing(self):
        other = make_archive(self.tmp, "something-else", {"README.md": "nope\n"})
        result = self.install(archive=other)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not Neat GenAI Studio", result.stderr)
        self.assertFalse((self.work / "neat-genai-studio").exists())


if __name__ == "__main__":
    unittest.main()
