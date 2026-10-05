"""A candidate waits for its own execution before capturing or releasing it."""

from __future__ import annotations

import asyncio
import subprocess

import httpx
import openai
import pytest

from opencollab import OpenCollab
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.adapters.env import LocalEnvironment
from opencollab.application.workflow import WorkflowContext
from opencollab.bootstrap import _workflow_runtime_session as runtime
from tests.support.workflow_context_test_support import FakeFactory, FakeSession


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCOLLAB_UNBOUNDED_LIMITS", raising=False)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    (tmp_path / "source.txt").write_text("initial\n")
    for args in (
        ("init", "-q"),
        ("add", "source.txt"),
        ("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-qm", "\u521d\u59cb\u5316\u6d4b\u8bd5\u4ed3\u5e93"),
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture
def provider_calls(monkeypatch):
    original_client = openai.AsyncOpenAI
    calls = []

    async def respond(request):
        calls.append(request)
        return httpx.Response(200, json={
            "id": "test-response", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def build_client(**kwargs):
        return original_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "AsyncOpenAI", build_client)
    return calls


def _client(repository):
    return OpenCollab(
        repository, model="test-model", provider="openai",
        api_key="test-key",  # pragma: allowlist secret -- in-memory transport
        base_url="https://provider.example.invalid/v1", config={"llm_max_retries": 0},
    )


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_public_candidate_and_ordinary_agent_complete(repository, provider_calls, concurrency):
    async def workflow(ctx, _args):
        values = await ctx.parallel([
            lambda: ctx.candidate_agent("candidate", label="candidate"),
            lambda: ctx.agent("ordinary", label="ordinary"),
        ])
        return [values[0].output, values[1]]

    task = asyncio.create_task(_client(repository).workflow(
        workflow, budget=100_000, concurrency=concurrency, task_concurrency=2, timeout=None, trace=False,
    ))
    try:
        done, _ = await asyncio.wait({task}, timeout=3)
        assert done, "candidate completed its model call but blocked the queued ordinary agent"
        result = task.result()
        assert result.ok, result.reason
        assert result.output == ["answer", "answer"]
        assert len(provider_calls) == 2
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_public_candidate_waits_for_session_subscriber_write(repository, provider_calls, monkeypatch):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    candidate_returned = asyncio.Event()
    context = None
    original_build = runtime.build_session

    class DelayedWriteSubscriber:
        def __init__(self, environment):
            self.environment = environment
            self.pending_tasks = []

        async def write(self):
            cleanup_started.set()
            await release_cleanup.wait()
            await self.environment.write_file("source.txt", "late candidate write\n")

        def emit(self, event):
            if event.type == "step_end" and not self.pending_tasks:
                self.pending_tasks.append(asyncio.create_task(self.write()))

    def build(**kwargs):
        session = original_build(**kwargs)
        session.event_bus.subscribe(DelayedWriteSubscriber(kwargs["env"]))
        return session

    monkeypatch.setattr(runtime, "build_session", build)

    async def workflow(ctx, _args):
        nonlocal context
        context = ctx
        candidate = await ctx.candidate_agent("candidate", label="candidate", budget=100_000)
        candidate_returned.set()
        return candidate.diff

    task = asyncio.create_task(_client(repository).workflow(
        workflow, budget=100_000, concurrency=1, timeout=None, trace=False,
    ))
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        await asyncio.sleep(0.02)
        assert not candidate_returned.is_set()
        assert context.tokens_remaining() == 0
        assert context._semaphore.locked()
        assert (repository / "source.txt").read_text() == "initial\n"
    finally:
        release_cleanup.set()
        result = await asyncio.wait_for(task, timeout=5)

    assert result.ok, result.reason
    assert "+late candidate write" in result.output
    assert len(provider_calls) == 1
    assert context.tokens_remaining() == 99_985
    assert (repository / "source.txt").read_text() == "initial\n"


@pytest.mark.parametrize("ending", ["timeout", "cancel", "cancel-twice"])
async def test_candidate_keeps_lease_for_cleanup_spawned_by_cancelled_turn(repository, ending):
    started = asyncio.Event()
    cancel_seen = asyncio.Event()
    release_turn = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    later_entered = asyncio.Event()

    class LateCleanupSession(FakeSession):
        pending_cleanup_tasks = ()

        async def write(self):
            cleanup_started.set()
            await release_cleanup.wait()
            await self.environment.write_file("source.txt", "late candidate write\n")

        async def run_loop(self, cancel_event=None):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                await release_turn.wait()
                self.pending_cleanup_tasks = (asyncio.create_task(self.write()),)
                raise

    slow = LateCleanupSession(tokens=7)

    async def enter_later():
        later_entered.set()

    class Factory(FakeFactory):
        def build_workflow_session(self, **kwargs):
            session = super().build_workflow_session(**kwargs)
            session.environment = kwargs["env"]
            return session

    context = WorkflowContext(
        Factory([slow, FakeSession(reply="later", on_enter=enter_later)]),
        max_concurrency=1, budget_total=100,
        candidate_workspace=EnvCandidateWorkspace(LocalEnvironment(str(repository))),
    )
    task = asyncio.create_task(context.candidate_agent(
        "candidate", label="candidate", budget=90, timeout=0.1 if ending == "timeout" else None,
    ))
    later_task = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if ending != "timeout":
            task.cancel()
        await asyncio.wait_for(cancel_seen.wait(), timeout=5)
        release_turn.set()
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        if ending == "cancel-twice":
            task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert context.tokens_remaining() == 10
        later_task = asyncio.create_task(context.agent("ordinary", budget=1))
        await asyncio.sleep(0.02)
        assert not later_entered.is_set()
        assert (repository / "source.txt").read_text() == "initial\n"
    finally:
        release_turn.set()
        release_cleanup.set()
        if ending != "timeout":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        else:
            candidate = await asyncio.wait_for(task, timeout=5)
            assert candidate.output is None
            assert "+late candidate write" in candidate.diff
        if later_task is not None:
            assert await asyncio.wait_for(later_task, timeout=5) == "later"
        await asyncio.gather(*slow.pending_cleanup_tasks, return_exceptions=True)

    assert context.tokens_remaining() == 93
    assert context.pending_cleanup_tasks == ()
    assert (repository / "source.txt").read_text() == "initial\n"
