from __future__ import annotations

from pathlib import Path

import pytest

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


def _dependency(tmp_path: Path, name: str) -> Path:
    parent = tmp_path / name
    parent.mkdir()
    return _repository(parent)


@pytest.mark.parametrize("nested", [False, True])
async def test_candidate_reads_committed_source_available_submodules(tmp_path, monkeypatch, nested):
    dependency = _dependency(tmp_path, "dependency")
    dependency_path = "vendor/dependency"
    if nested:
        middle = _dependency(tmp_path, "middle")
        _git(middle, "-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor/leaf")
        _git(middle, "config", "-f", ".gitmodules", "submodule.vendor/leaf.url", "https://example.invalid/leaf.git")
        _git(middle, "add", ".gitmodules", "vendor/leaf")
        _git(middle, "commit", "-m", "nested dependency")
        dependency = middle
        dependency_path += "/vendor/leaf"
    source = _repository(tmp_path)
    _git(source, "-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor/dependency")
    if nested:
        _git(
            source / "vendor/dependency", "-c", "protocol.file.allow=always",
            "-c", f"submodule.vendor/leaf.url={tmp_path / 'dependency/repo'}",
            "submodule", "update", "--init", "--no-fetch", "--", "vendor/leaf",
        )
    _git(source, "config", "-f", ".gitmodules", "submodule.vendor/dependency.url", "https://example.invalid/dep.git")
    _git(source, "add", ".gitmodules", "vendor/dependency")
    _git(source, "commit", "-m", "committed dependency")
    assert (source / dependency_path / "source.py").read_text() == "value = 1\n"
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    base = LocalEnvironment(str(source))
    lease = None
    try:
        lease = await EnvCandidateWorkspace(base).acquire("dependency")
        assert await lease.environment.read_file(f"{dependency_path}/source.py") == "value = 1\n"
        assert await lease.diff() == ""
        candidate_path = Path(lease.candidate_workspace)
    finally:
        if lease is not None:
            await lease.cleanup()
        await base.cleanup()
    assert not candidate_path.exists()
    assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1
