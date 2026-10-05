"""Real session and scheduler accounting for providers that outlive cancellation."""

from __future__ import annotations

import asyncio

import pytest

from opencollab.adapters.storage import SessionStore
from opencollab.adapters.tools.spawn import SpawnAgentTool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler, SchedulerTurnError
from opencollab.application.session import SessionBusyError
from opencollab.application.session_run import GenerationTimeoutError
from opencollab.bootstrap import build_session
from opencollab.domain.session import SessionPhase
from tests.support.session_characterization_test_support import (
    FakeAgent,
    FakeLLMClient,
    fake_worktree_pool,
    llm_response,
    tool_call,
)


class _DelayedProvider:
    def __init__(self, outcome="success"):
        self.outcome = outcome
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_count = 0

    async def complete(self, **kwargs):
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancel_count += 1
                self.cancelled.set()
                if self.outcome == "cancelled":
                    raise
        if self.outcome == "error":
            raise RuntimeError("late provider failure")
        return llm_response(content="provider response", input_tokens=13, output_tokens=24)


async def _drain(session):
    while session.pending_cleanup_tasks or session.runner.pending_cleanup_tasks:
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("trigger", "outcome", "repeat_cancel"),
    [
        ("cancel", "success", False),
        ("cancel", "success", True),
        ("timeout", "success", False),
        ("timeout", "success", True),
        ("cancel", "error", False),
        ("timeout", "error", True),
        ("cancel", "cancelled", False),
        ("timeout", "cancelled", False),
    ],
)
async def test_session_accounts_only_successful_abandoned_response_once(
    trigger, outcome, repeat_cancel, tmp_path,
):
    provider = _DelayedProvider(outcome)
    session = build_session(
        agent=FakeAgent(), llm=provider, llm_timeout=0.01 if trigger == "timeout" else 600,
    )
    await session.add_user_message("run")
    call = asyncio.create_task(session.run_loop())
    try:
        await asyncio.wait_for(provider.started.wait(), 1)
        if trigger == "cancel":
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call, 1)
            expected_phase = SessionPhase.STOPPED
        else:
            with pytest.raises(GenerationTimeoutError):
                await asyncio.wait_for(call, 1)
            expected_phase = SessionPhase.ERROR
        await asyncio.wait_for(provider.cancelled.wait(), 1)
        assert session.phase is expected_phase
        assert session.used_tokens == 0

        if outcome != "cancelled":
            pending = session.pending_cleanup_tasks
            assert len(pending) == 1
            with pytest.raises(SessionBusyError):
                await session.add_user_message("next turn")
            with pytest.raises(SessionBusyError):
                session.restore(str(tmp_path / "unread.json"))
            if repeat_cancel:
                pending[0].cancel()
                await asyncio.sleep(0)
                assert provider.cancel_count == 2
                assert session.pending_cleanup_tasks == pending

        provider.release.set()
        await asyncio.wait_for(_drain(session), 1)
        expected_usage = 37 if outcome == "success" else 0
        assert session.used_tokens == expected_usage
        assert session.runner.late_provider_usage == ((37,) if outcome == "success" else ())
        assert session.phase is expected_phase
        assert all(message.get("content") != "provider response" for message in session.messages)
        assert session.pending_cleanup_tasks == ()
        assert session.runner.pending_cleanup_tasks == ()

        path = tmp_path / "drained.json"
        session.save(str(path))
        snapshot = SessionStore().load_snapshot(str(path), session.agent.system_prompt)
        assert snapshot["session_state"]["used_tokens"] == expected_usage
        provider.outcome = "success"
        await session.add_user_message("next turn")
        assert await session.run_loop() == "provider response"
        assert session.used_tokens == expected_usage + 37
        assert session.runner.late_provider_usage == ((37,) if outcome == "success" else ())
    finally:
        provider.release.set()
        await asyncio.gather(call, return_exceptions=True)
        await _drain(session)


async def test_session_accounts_provider_success_when_cancellation_races_with_completion():
    class Provider:
        async def complete(self, **kwargs):
            call.cancel()
            return llm_response(content="racing response", input_tokens=13, output_tokens=24)

    session = build_session(agent=FakeAgent(), llm=Provider())
    await session.add_user_message("run")
    call = asyncio.create_task(session.run_loop())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, 1)
    await asyncio.wait_for(_drain(session), 1)

    assert session.phase is SessionPhase.STOPPED
    assert session.used_tokens == 37
    assert session.runner.late_provider_usage == (37,)
    assert all(message.get("content") != "racing response" for message in session.messages)


class _ChildFactory:
    def __init__(self, provider):
        self.provider = provider
        self.session = None

    def build_spawn_session(self, **kwargs):
        agent = FakeAgent()
        agent.name = kwargs["role"]
        self.session = build_session(
            agent=agent, llm=self.provider, aid=kwargs["aid"], max_budget_tokens=kwargs["budget"],
        )
        return self.session


@pytest.mark.parametrize("cancel", [False, True])
async def test_scheduler_preserves_child_usage_after_parent_cancellation(cancel, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = _DelayedProvider()
    factory = _ChildFactory(provider)
    scheduler = Scheduler(
        session_factory=factory, worktree_pool=fake_worktree_pool(), event_sink=EventBus(),
    )
    lead_model = FakeLLMClient([
        llm_response(tool_calls=[tool_call(
            "delegation", "spawn_agent", '{"role":"coder","task":"controlled work"}',
        )], finish_reason="tool_calls"),
        llm_response(content="lead response"),
    ])
    lead = build_session(agent=FakeAgent(tools=[SpawnAgentTool(scheduler)]), llm=lead_model)
    scheduler.register_lead(lead)
    cancel_event = asyncio.Event()
    turn = asyncio.create_task(scheduler.run("run", cancel_event=cancel_event))
    try:
        await asyncio.wait_for(provider.started.wait(), 1)

        async def wait_for_suspension():
            while lead.phase is not SessionPhase.AWAITING_EVENTS:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_suspension(), 1)
        if cancel:
            cancel_event.set()
            with pytest.raises(SchedulerTurnError, match="interrupted by user"):
                await asyncio.wait_for(turn, 1)
            await asyncio.wait_for(provider.cancelled.wait(), 1)
            assert factory.session.pending_cleanup_tasks
            assert factory.session.used_tokens == 0
        provider.release.set()
        if cancel:
            await asyncio.wait_for(_drain(factory.session), 1)
            assert factory.session.phase is SessionPhase.STOPPED
            assert factory.session.runner.late_provider_usage == (37,)
            assert all(message.get("content") != "provider response" for message in factory.session.messages)
        else:
            assert await asyncio.wait_for(turn, 1) == "lead response"
            assert factory.session.runner.late_provider_usage == ()
        assert factory.session.used_tokens == 37
        assert scheduler.table.get(1).state.used_tokens == 37
        assert scheduler.used_tokens == lead.used_tokens + 37
    finally:
        provider.release.set()
        await asyncio.gather(turn, return_exceptions=True)
        await _drain(factory.session)
        await scheduler.cleanup()
