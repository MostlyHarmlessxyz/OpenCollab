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
