from __future__ import annotations

from pathlib import Path

import pytest

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("mixed_text", [False, True])
async def test_candidate_inherits_captures_and_adopts_binary_contents(tmp_path, tracked, mixed_text):
    repo = _repository(tmp_path)
    binary_path = repo / "zbytes.bin"
    if tracked:
        binary_path.write_bytes(b"committed\x00\xfd\n")
        _git(repo, "add", "zbytes.bin")
        _git(repo, "commit", "-m", "committed binary")
    source_bytes = b"source\x00\xff\n"
    candidate_bytes = b"candidate\x00\xfe\n"
    binary_path.write_bytes(source_bytes)
    if mixed_text:
        (repo / "source.py").write_text("value = 2\n")
        (repo / "draft.txt").write_text("source draft\n")
    source_index = _git(repo, "ls-files", "--stage")
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    lease = None
    try:
        lease = await workspace.acquire("binary-contents")
        candidate = Path(lease.candidate_workspace)
        assert (candidate / "zbytes.bin").read_bytes() == source_bytes
        if mixed_text:
            assert await lease.environment.read_file("source.py") == "value = 2\n"
            assert await lease.environment.read_file("draft.txt") == "source draft\n"
        assert await lease.diff() == ""
        (candidate / "zbytes.bin").write_bytes(candidate_bytes)
        if mixed_text:
            await lease.environment.write_file("source.py", "value = 3\n")
            await lease.environment.write_file("draft.txt", "candidate draft\n")
        patch = await lease.diff()
        assert "GIT binary patch" in patch
        assert binary_path.read_bytes() == source_bytes
        await workspace.adopt(patch)
        assert binary_path.read_bytes() == candidate_bytes
        if mixed_text:
            assert (repo / "source.py").read_text() == "value = 3\n"
            assert (repo / "draft.txt").read_text() == "candidate draft\n"
        assert _git(repo, "ls-files", "--stage") == source_index
    finally:
        if lease is not None:
            await lease.cleanup()
        await base.cleanup()
