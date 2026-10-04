from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace, _raw_diff_at
from tests.workflows.test_workflow_candidate_workspace import _repository


@pytest.mark.parametrize("index_file", [None, "/tmp/candidate-index"])
async def test_candidate_diff_commands_disable_display_conversion(index_file):
    commands = []

    class Environment:
        async def exec_cmd(self, command, **_kwargs):
            commands.append(shlex.split(command))
            if "ls-files" in commands[-1]:
                return ExecResult(0, "draft.txt\0", "")
            return ExecResult(0, "", "")

    assert await _raw_diff_at(Environment(), "/workspace", index_file=index_file) == ""
    diff_commands = [command for command in commands if "diff" in command]
    assert len(diff_commands) == 2
    assert all("--no-textconv" in command for command in diff_commands)
    assert all("--no-ext-diff" in command for command in diff_commands)


async def test_candidate_inherits_and_delivers_raw_source_contents(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    (repo / "source.py").write_text("value = 2\n")
    (repo / "bytes.bin").write_bytes(b"source\x00\xff\n")
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    execute = base.exec_cmd

    async def display_conversion(command, **kwargs):
        # Model a display converter that maps all tracked contents to the same
        # text. This seam records commands without installing a Git driver.
        arguments = shlex.split(command)
        if "diff" in arguments and "--no-textconv" not in arguments:
            return ExecResult(0, "", "")
        return await execute(command, **kwargs)

    monkeypatch.setattr(base, "exec_cmd", display_conversion)
    lease = None
    try:
        source_patch = await workspace.source_diff()
        assert "+value = 2" in source_patch
        assert "GIT binary patch" in source_patch
        lease = await workspace.acquire("raw-contents")
        candidate = Path(lease.candidate_workspace)
        assert await lease.environment.read_file("source.py") == "value = 2\n"
        assert (candidate / "bytes.bin").read_bytes() == b"source\x00\xff\n"
        await lease.environment.write_file("source.py", "value = 3\n")
        (candidate / "bytes.bin").write_bytes(b"candidate\x00\xfe\n")
        patch = await lease.diff()
        assert "-value = 2" in patch
        assert "+value = 3" in patch
        assert "GIT binary patch" in patch
        await workspace.adopt(patch)
        assert (repo / "source.py").read_text() == "value = 3\n"
        assert (repo / "bytes.bin").read_bytes() == b"candidate\x00\xfe\n"
    finally:
        if lease is not None:
            await lease.cleanup()
        await base.cleanup()
