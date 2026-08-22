"""Guards against the repo silently losing source, or silently gaining data.

Both failure modes are quiet: `git add` prints a hint and carries on, so a
gitignore that is too broad drops files from a commit without failing anything.
"""

import shutil
import subprocess

import pytest

from src.utils.paths import repo_root

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=False
    )


def _is_ignored(relative_path: str) -> bool:
    return _git("check-ignore", "-q", relative_path).returncode == 0


@needs_git
def test_no_source_package_is_gitignored():
    """`data/` in .gitignore matched src/data/ as well as the dataset directory,
    so an entire package was dropped from a commit without any error."""
    packages = sorted(p.parent for p in (repo_root() / "src").rglob("__init__.py"))
    assert packages, "expected at least one package under src/"

    ignored = [
        pkg.relative_to(repo_root()).as_posix()
        for pkg in packages
        if _is_ignored(pkg.relative_to(repo_root()).as_posix())
    ]
    assert not ignored, f"these source packages are gitignored: {ignored}"


@needs_git
@pytest.mark.parametrize("path", ["src", "scripts", "tests", "configs", "docs", "notebooks"])
def test_project_directories_are_not_ignored(path):
    assert not _is_ignored(path)


@needs_git
@pytest.mark.parametrize("path", ["data", "results/runs", "results/validation"])
def test_artifact_directories_stay_ignored(path):
    """The flip side: anchoring the rules must not stop them ignoring the real
    dataset and results directories."""
    assert _is_ignored(path)


@needs_git
def test_split_manifests_are_committed_not_ignored():
    """Everything under results/ is ignored except the manifests, which are the
    reproducibility record and have to be in the repo."""
    assert not _is_ignored("results/manifests/splits")


@needs_git
def test_no_dataset_or_checkpoint_files_are_tracked():
    tracked = _git("ls-files").stdout.split()
    bad = [
        f
        for f in tracked
        if f.startswith("data/")
        or f.endswith((".npy", ".npz", ".pkl", ".pt", ".pth", ".ckpt", ".onnx", ".safetensors"))
    ]
    assert not bad, f"dataset or checkpoint files are tracked: {bad}"
