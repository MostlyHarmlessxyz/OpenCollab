from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import (
    CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
    EnvCandidateWorkspace,
    _raw_diff_at,
)
from opencollab.application.workflow import WorkflowContext
from opencollab.application.workflow_candidates import (
    CandidateWorkspaceTrackingError,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "source.py").write_text("value = 1\n")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "initial")
    return repo


class _RecordingDiffEnvironment:
    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.results = [
            ExecResult(0, "", ""),
            ExecResult(0, "", ""),
        ]

    async def exec_cmd(self, _command: str, timeout: float = 120.0) -> ExecResult:
        self.timeouts.append(timeout)
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_candidate_source_diff_uses_pre_model_workspace_timeout() -> None:
    environment = _RecordingDiffEnvironment()

    assert await _raw_diff_at(environment, "/app") == ""
    assert environment.timeouts == [CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS] * 2


class _EditingSession:
    used_tokens = 7
    pending_cleanup_tasks: tuple[Any, ...] = ()

    def __init__(self, environment: Any, *, fail: bool = False) -> None:
        self.environment = environment
        self.fail = fail
        self.state = type("State", (), {"messages": []})()

    async def add_user_message(self, content: str) -> None:
        self.state.messages.append({"role": "user", "content": content})

    async def run_loop(self, _cancel_event: Any = None) -> str:
        await self.environment.write_file("source.py", "value = 2\n")
        if self.fail:
            raise RuntimeError("role failed after writing")
        return "done"


class _Factory:
    def __init__(self, fallback: Any, *, fail: bool = False, ignore_override: bool = False):
        self.fallback = fallback
        self.fail = fail
        self.ignore_override = ignore_override
        self.environments: list[Any] = []

    def build_workflow_session(self, **kwargs: Any) -> _EditingSession:
        environment = self.fallback if self.ignore_override else kwargs.get("env")
        if environment is None:
            environment = self.fallback
        self.environments.append(environment)
        return _EditingSession(environment, fail=self.fail)


@pytest.mark.asyncio
async def test_candidate_agent_writes_only_candidate_worktree(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    factory = _Factory(base)
    ctx = WorkflowContext(
        factory,
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    candidate = await ctx.candidate_agent("edit", label="candidate-a")

    assert "value = 2" in candidate.diff
    assert (repo / "source.py").read_text() == "value = 1\n"
    assert factory.environments[0].workspace != str(repo)


@pytest.mark.asyncio
async def test_candidate_agent_preserves_partial_diff_after_role_error(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    ctx = WorkflowContext(
        _Factory(base, fail=True),
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    candidate = await ctx.candidate_agent("edit then fail", label="candidate-a")

    assert candidate.output is None
    assert "value = 2" in candidate.diff
    assert (repo / "source.py").read_text() == "value = 1\n"


@pytest.mark.asyncio
async def test_candidate_workflow_binds_nested_agent_to_candidate(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    factory = _Factory(base)
    ctx = WorkflowContext(
        factory,
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        return await child.agent("nested edit")

    candidate = await ctx.candidate_workflow(
        nested,
        {},
        label="candidate-workflow",
        budget=None,
    )

    assert "value = 2" in candidate.diff
    assert (repo / "source.py").read_text() == "value = 1\n"
    assert factory.environments[0].workspace != str(repo)


@pytest.mark.asyncio
async def test_source_worktree_leak_is_restored_and_reported(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    ctx = WorkflowContext(
        _Factory(base, ignore_override=True),
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    with pytest.raises(CandidateWorkspaceTrackingError):
        await ctx.candidate_agent("leak", label="candidate-a")

    assert (repo / "source.py").read_text() == "value = 1\n"
    assert _git(repo, "status", "--short") == ""


@pytest.mark.asyncio
async def test_base_environment_captures_relative_untracked_candidate_diff(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    lease = await workspace.acquire("candidate-a")
    try:
        await lease.environment.write_file("new_file.py", "created = True\n")
        diff = await lease.diff()
    finally:
        await lease.cleanup()

    assert "b/new_file.py" in diff
    assert lease.candidate_workspace not in diff


@pytest.mark.asyncio
async def test_candidate_isolated_role_reads_current_candidate_and_owns_cleanup(tmp_path, monkeypatch):
    from opencollab.bootstrap import _workflow_runtime_session as runtime
    from tests.support.workflow_context_test_support import FakeSession

    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    built = []

    class ReadingSession(FakeSession):
        def __init__(self, environment):
            super().__init__(tokens=3)
            self.environment = environment
            self.closes = 0

        async def run_loop(self, cancel_event=None):
            if self.prompt == "edit candidate":
                await self.environment.write_file("source.py", "value = 2\n")
                await self.environment.write_file("untracked.txt", "candidate content\n")
                return "edited"
            value = await self.environment.read_file("source.py")
            extra = await self.environment.read_file("untracked.txt")
            await self.environment.write_file("source.py", "value = 9\n")
            return value + extra

        async def aclose(self):
            self.closes += 1

    def build(**kwargs):
        session = ReadingSession(kwargs["env"])
        built.append(session)
        return session

    monkeypatch.setattr(runtime, "build_session", build)
    factory = runtime.WorkflowSessionFactory(
        model="gpt-fake", provider="openai", api_key=None, base_url=None,
        workspace=str(repo), env=base,
    )
    parent = WorkflowContext(factory, budget_total=100, candidate_workspace=EnvCandidateWorkspace(base))

    async def nested(child, _args):
        await child.agent("edit candidate")
        reply = await child.agent("inspect candidate", isolation=True, label="scout")
        assert await built[0].environment.read_file("source.py") == "value = 2\n"
        return reply

    candidate = await parent.candidate_workflow(nested, {}, label="candidate")
    assert candidate.output == "value = 2\ncandidate content\n"
    assert parent.tokens_spent() == 6
    assert len(parent.sessions) == 2
    assert built[1].environment.workspace != built[0].environment.workspace
    assert built[1].closes == 1
    assert built[0].closes == 0
    assert base.revoked is False
    assert (repo / "source.py").read_text() == "value = 1\n"
    assert len(_git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2
    await factory.release_isolated_envs()
    assert base.revoked is False


@pytest.mark.parametrize("source_edit", [False, True])
@pytest.mark.asyncio
async def test_candidate_source_changed_honors_excluded_paths(tmp_path, source_edit):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    factory = _Factory(base)
    parent = WorkflowContext(factory, candidate_workspace=EnvCandidateWorkspace(base))

    async def nested(child, _args):
        environment = child._factory._environment
        await environment.write_file("generated_test.py", "test_case = 1\n")
        if source_edit:
            await environment.write_file("source.py", "value = 2\n")
        return {"all": await child.tree_changed(), "source": await child.source_changed(["generated_test.py"])}

    candidate = await parent.candidate_workflow(nested, {}, label="candidate")
    assert candidate.output == {"all": True, "source": source_edit}
    assert "generated_test.py" in candidate.diff
