"""Public waiting-turn persistence with real tools and local fake completions."""

from __future__ import annotations

import asyncio

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.storage import SessionStore
from opencollab.adapters.tools.spawn import SpawnAgentTool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from opencollab.bootstrap import build_session, load_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from tests.support.session_characterization_test_support import FakeLLMClient, llm_response, tool_call


def agent(tools=()):
    return Agent(name="lead", system_prompt="Finish the task.", tools=list(tools))


class DelayedClient(FakeLLMClient):
    async def complete(self, **kwargs):
        await asyncio.sleep(0.03)
        return await super().complete(**kwargs)


class ChildClient:
    def __init__(self):
        self.release = asyncio.Event()

    async def complete(self, **kwargs):
        await self.release.wait()
        return llm_response(content="child done")


class Pool:
    def __init__(self, workspace):
        self.env = LocalEnvironment(str(workspace))

    async def acquire(self, role):
        return self.env

    async def release(self):
        pass


class Factory:
    def __init__(self, workspace):
        self.workspace = workspace
        self.client = ChildClient()

    def build_spawn_session(self, **kwargs):
        return build_session(
            agent=agent(), llm=self.client, env=LocalEnvironment(str(self.workspace)),
            max_budget_tokens=kwargs["budget"], aid=kwargs["aid"],
        )


def make_team(workspace, *, extra_calls=(), extra_tools=(), autosave=True):
    factory = Factory(workspace)
    scheduler = Scheduler(session_factory=factory, worktree_pool=Pool(workspace), event_sink=EventBus())
    calls = [tool_call("child", "spawn_agent", '{"role":"worker","task":"old child"}'), *extra_calls]
    model = DelayedClient([
        llm_response(tool_calls=calls), llm_response(content="old task answer"),
    ])
    events = []
    path = workspace / "waiting.json"
    session = build_session(
        agent=agent([SpawnAgentTool(scheduler), *extra_tools]), llm=model,
        env=LocalEnvironment(str(workspace)), auto_save_path=str(path) if autosave else None,
        event_sink=EventBus(events.append),
    )
    scheduler.register_lead(session)
    return scheduler, factory, session, model, events, path


async def wait_for_suspension(session):
    async with asyncio.timeout(2):
        while session.phase is not SessionPhase.AWAITING_EVENTS:
            await asyncio.sleep(0.001)
    await asyncio.gather(*session.pending_cleanup_tasks)


async def finish_team(scheduler, factory, active):
    factory.client.release.set()
    if not active.done():
        await asyncio.wait_for(asyncio.shield(active), 2)
    await scheduler.cleanup()


@pytest.mark.asyncio
async def test_awaiting_autosave_restores_old_turn_before_accepting_new_request(tmp_path):
    scheduler, factory, session, model, events, path = make_team(tmp_path)
    active = asyncio.create_task(scheduler.run("old user task"))
    resumed_scheduler = None
    try:
        await wait_for_suspension(session)
        snapshot = SessionStore().load_snapshot(str(path), session.agent.system_prompt)
        assert snapshot["session_state"]["phase"] == "awaiting_events"
        assert snapshot["session_state"]["pending_events"][0]["ref"] == 1
        latency = snapshot["session_state"]["pending_step_latency"]
        assert latency >= 0.03
        assert not [event for event in events if event.type == "step_end"]

        resumed_model = FakeLLMClient([
            llm_response(content="old resumed answer"), llm_response(content="new task answer"),
        ])
        restored = load_session(
            str(path), agent=agent(), llm=resumed_model, env=LocalEnvironment(str(tmp_path)),
        )
        assert restored.phase is SessionPhase.AWAITING_EVENTS
        resumed_scheduler = Scheduler(
            session_factory=factory, worktree_pool=Pool(tmp_path), event_sink=EventBus(),
        )
        resumed_scheduler.register_lead(restored)

        assert await resumed_scheduler.run("new user task") == "new task answer"
        assert len(resumed_model.calls) == 2
        assert all("new user task" not in str(message.get("content"))
                   for message in resumed_model.calls[0]["messages"])
        assert any(message.get("content") == "old resumed answer"
                   for message in resumed_model.calls[1]["messages"])

        factory.client.release.set()
        assert await asyncio.wait_for(active, 2) == "old task answer"
        completed = [event for event in events if event.type == "step_end" and event.data["step"] == 1]
        assert len(completed) == 1
        assert completed[0].data["latency"] == latency
        assert len(model.calls) == 2
    finally:
        await finish_team(scheduler, factory, active)
        if resumed_scheduler is not None:
            await resumed_scheduler.cleanup()
