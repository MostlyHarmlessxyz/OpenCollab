"""Message append ownership prevents an external turn from starting another driver."""

import asyncio

import pytest

from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.bootstrap import build_session
from opencollab.domain.session import SessionPhase
from tests.support.session_characterization_test_support import (
    FakeAgent,
    FakeLLMClient,
    fake_team_session_factory,
    fake_worktree_pool,
    llm_response,
)


@pytest.mark.asyncio
async def test_external_turn_respects_existing_message_delivery_owner():
    lead = build_session(agent=FakeAgent(), llm=FakeLLMClient([llm_response("unexpected external answer")]))
    scheduler = Scheduler(
        session_factory=fake_team_session_factory()[1],
        worktree_pool=fake_worktree_pool(), event_sink=EventBus(),
    )
    scheduler.register_lead(lead)
    release = asyncio.Event()
    owner = asyncio.create_task(release.wait())
    scheduler._message_delivery_tasks[0] = owner
    lead.state.set_phase(SessionPhase.DONE)
    before = list(lead.messages)
    try:
        with pytest.raises(RuntimeError, match="message delivery"):
            await scheduler.run_turn(0, "external request")
        assert lead.phase is SessionPhase.DONE
        assert lead.messages == before
        assert scheduler._tasks.get(0) is None
    finally:
        release.set()
        await owner
        scheduler._message_delivery_tasks.pop(0, None)
        await scheduler.cleanup()


class _ReplyProvider:
    def __init__(self, release=None):
        self.release = release
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append([dict(message) for message in kwargs["messages"]])
        if self.release is not None:
            await self.release.wait()
        return llm_response(f"answer-{len(self.calls)}")


class _PassiveWorktreePool:
    async def acquire(self, role):
        return None

    async def release(self):
        pass


class _MessageSessionFactory:
    def __init__(self, release):
        self.release = release

    def build_spawn_session(self, **kwargs):
        from opencollab.adapters.tools.message import MessageAgentTool
        from opencollab.domain.agent import Agent

        return build_session(
            agent=Agent(
                name=kwargs["role"],
                system_prompt="Answer the task.",
                tools=[MessageAgentTool(kwargs["scheduler"])],
            ),
            llm=_ReplyProvider(self.release),
            aid=kwargs["aid"],
            seed_user_messages=[{"role": "user", "content": kwargs["task"]}],
        )


async def _real_message_scheduler(tmp_path, *, max_steps=100):
    from opencollab.domain.agent import Agent

    release = asyncio.Event()
    provider = _ReplyProvider()
    lead = build_session(
        agent=Agent(name="lead", system_prompt="Answer the question."),
        llm=provider,
        auto_save_path=str(tmp_path / "lead.json"),
        max_steps=max_steps,
    )
    scheduler = Scheduler(
        session_factory=_MessageSessionFactory(release),
        worktree_pool=_PassiveWorktreePool(),
        event_sink=EventBus(),
    )
    scheduler.register_lead(lead)
    sender = await scheduler.spawn(0, "worker", "Prepare an update.")
    return scheduler, lead, sender, release, provider


async def _wait_for_external_append(lead):
    for _ in range(100):
        if lead.state.pending_external_user_turn is not None:
            return
        await asyncio.sleep(0)
    raise AssertionError("external user append did not start")


async def _message_tool_send(scheduler, sender, content):
    from opencollab.adapters.tools.message import MessageAgentTool
    from opencollab.application.tool_execution import ToolRuntime

    tool = scheduler._sessions[sender].agent.find_tool("message_agent")
    assert isinstance(tool, MessageAgentTool)
    return await tool.execute_with_runtime(
        {"to_aid": 0, "summary": content, "content": content},
        ToolRuntime(environment=None, safety_policy=None, permission_policy=None, aid=sender),
    )


def _delivered_updates(messages):
    return [
        message["content"]
        for message in messages
        if message.get("role") == "user" and "<teammate-message" in message.get("content", "")
    ]


async def test_message_tool_queues_during_default_autosave_append_and_delivers_fifo(tmp_path):
    scheduler, lead, sender, release, provider = await _real_message_scheduler(tmp_path)
    active = asyncio.create_task(scheduler.run("external question"))
    try:
        await _wait_for_external_append(lead)
        owner = scheduler._message_delivery_tasks[0]
        assert owner is active
        for content in ("first update", "second update"):
            ack = await _message_tool_send(scheduler, sender, content)
            assert ack.startswith("Message queued to aid 0.")
            assert scheduler._message_delivery_tasks[0] is owner
        assert [message["summary"] for message in lead.state.pending_user_messages] == [
            "first update",
            "second update",
        ]
        release.set()
        assert await asyncio.wait_for(active, 2) == "answer-2"
        assert len(provider.calls) == 2
        assert "external question" in provider.calls[0][-1]["content"]
        delivered = _delivered_updates(lead.messages)
        assert len(delivered) == 1
        assert delivered[0].count("first update") == 2  # summary and body
        assert delivered[0].count("second update") == 2
        assert delivered[0].index("first update") < delivered[0].index("second update")
        assert not lead.state.pending_user_messages
        assert not scheduler._message_inbox[0]
        await scheduler.cleanup()
        durable = lead.store.load_snapshot(lead.auto_save_path, lead.agent.system_prompt)
        assert _delivered_updates(durable["messages"]) == delivered
        assert not durable.get("pending_messages")
    finally:
        release.set()
        await scheduler.cleanup()
        await asyncio.gather(active, return_exceptions=True)


@pytest.mark.parametrize("ending", ["cancel", "step_limit"])
async def test_accepted_message_append_race_keeps_stopped_turn_explicit(tmp_path, ending):
    from opencollab.application.scheduler_types import SchedulerTurnError

    scheduler, lead, sender, release, provider = await _real_message_scheduler(
        tmp_path,
        max_steps=0 if ending == "step_limit" else 100,
    )
    cancel_event = asyncio.Event()
    active = asyncio.create_task(scheduler.run("external question", cancel_event=cancel_event))
    try:
        await _wait_for_external_append(lead)
        ack = await _message_tool_send(scheduler, sender, "accepted update")
        assert ack.startswith("Message queued to aid 0.")
        if ending == "cancel":
            cancel_event.set()
        release.set()
        with pytest.raises(SchedulerTurnError) as stopped:
            await asyncio.wait_for(active, 2)
        assert stopped.value.phase is SessionPhase.STOPPED
        assert lead.phase is SessionPhase.STOPPED
        assert not provider.calls
        assert len(_delivered_updates(lead.messages)) == 1
        assert not lead.state.pending_user_messages
        assert not scheduler._message_inbox[0]
        await scheduler.cleanup()
        durable = lead.store.load_snapshot(lead.auto_save_path, lead.agent.system_prompt)
        assert durable["session_state"]["phase"] == "stopped"
        assert not durable.get("pending_messages")
    finally:
        release.set()
        await scheduler.cleanup()
        await asyncio.gather(active, return_exceptions=True)


async def test_message_tool_preserves_real_append_failure_and_queued_message(tmp_path, monkeypatch):
    scheduler, lead, sender, release, provider = await _real_message_scheduler(tmp_path)

    def fail_append(*args, **kwargs):
        raise OSError("append failed")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(lead.state, "append_queued_external_user_turn", fail_append)
            with pytest.raises(OSError, match="append failed"):
                await _message_tool_send(scheduler, sender, "retained update")
        assert len(lead.state.pending_user_messages) == 1
        assert len(scheduler._message_inbox[0]) == 1
        assert scheduler._message_delivery_tasks == {}
        release.set()
        assert await asyncio.wait_for(scheduler.run("external question"), 2) == "answer-2"
        assert len(provider.calls) == 2
        assert len(_delivered_updates(lead.messages)) == 1
        assert not lead.state.pending_user_messages
        assert not scheduler._message_inbox[0]
    finally:
        release.set()
        await scheduler.cleanup()
