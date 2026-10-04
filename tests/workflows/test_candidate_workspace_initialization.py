from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

import pytest

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


@pytest.mark.parametrize("phase", ["add", "setup"])
async def test_cancelled_candidate_initialization_cleans_only_its_worktree(tmp_path, monkeypatch, phase):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    healthy = await workspace.acquire("healthy")
    entered = asyncio.Event()
    command_release = asyncio.Event()
    command_stopped = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    candidate_paths = []
    execute = base.exec_cmd
    create_environment = workspace._candidate_environment

    async def scheduled_command(command, **kwargs):
        arguments = shlex.split(command)
        if "worktree" in arguments and "add" in arguments:
            path = Path(arguments[-2])
            candidate_paths.append(path)
            result = await execute(command, **kwargs)
            if phase == "add":
                entered.set()
                try:
                    await command_release.wait()
                finally:
                    command_stopped.set()
            return result
        if "worktree" in arguments and "remove" in arguments:
            assert command_stopped.is_set()
            cleanup_entered.set()
            await cleanup_release.wait()
        return await execute(command, **kwargs)

    async def scheduled_environment(path):
        environment = await create_environment(path)
        if phase == "setup":
            original_setup = environment.setup

            async def setup():
                entered.set()
                try:
                    await command_release.wait()
                    return await original_setup()
                finally:
                    command_stopped.set()

            monkeypatch.setattr(environment, "setup", setup)
        return environment

    monkeypatch.setattr(base, "exec_cmd", scheduled_command)
    monkeypatch.setattr(workspace, "_candidate_environment", scheduled_environment)
    operation = asyncio.create_task(workspace.acquire("cancelled"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        operation.cancel()
        await asyncio.sleep(0)
        command_release.set()
        await asyncio.wait_for(cleanup_entered.wait(), 10)
        operation.cancel()
        await asyncio.sleep(0)
        assert candidate_paths[0].exists()
        assert Path(healthy.candidate_workspace).exists()
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 10)
        assert not candidate_paths[0].exists()
        assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 2
        assert await healthy.environment.read_file("source.py") == "value = 1\n"
        assert await base.read_file("source.py") == "value = 1\n"
    finally:
        command_release.set()
        cleanup_release.set()
        await asyncio.gather(operation, return_exceptions=True)
        monkeypatch.setattr(base, "exec_cmd", execute)
        for path in candidate_paths:
            if path.exists():
                _git(repo, "worktree", "remove", "--force", str(path))
        await healthy.cleanup()
        await base.cleanup()


@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_candidate_setup_failure_preserves_error_and_closes_environment(tmp_path, monkeypatch, cleanup_error):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    created = []
    failure = RuntimeError("candidate setup failed")
    create_environment = workspace._candidate_environment

    async def failing_environment(path):
        environment = await create_environment(path)
        created.append(environment)
        cleanup = environment.cleanup

        async def setup():
            raise failure

        async def failed_cleanup():
            await cleanup()
            raise OSError("candidate cleanup failed")

        monkeypatch.setattr(environment, "setup", setup)
        if cleanup_error:
            monkeypatch.setattr(environment, "cleanup", failed_cleanup)
        return environment

    monkeypatch.setattr(workspace, "_candidate_environment", failing_environment)
    try:
        with pytest.raises(RuntimeError) as caught:
            await workspace.acquire("setup-error")
        assert caught.value is failure
        assert created[0]._workspace_fd is None
        if cleanup_error:
            assert any("candidate cleanup failed" in note for note in getattr(failure, "__notes__", ()))
        else:
            assert not Path(created[0].workspace).exists()
            assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
        assert await base.read_file("source.py") == "value = 1\n"
    finally:
        for environment in created:
            await LocalEnvironment.cleanup(environment)
            if Path(environment.workspace).exists():
                _git(repo, "worktree", "remove", "--force", environment.workspace)
        await base.cleanup()
