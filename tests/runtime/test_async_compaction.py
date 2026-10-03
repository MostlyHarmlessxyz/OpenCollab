"""Default compaction cooperates with session deadlines and cancellation."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from opencollab.application.session_run import GenerationTimeoutError
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from tests.support.session_run_test_support import llm_response


def history(long=True):
    content = "x" * 18_000 if long else "earlier context"
    return [
        {"role": "system", "content": "Help the user."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Earlier result."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Another result."},
        {"role": "user", "content": "Now finish."},
    ]


class SlowModel:
    def __init__(self, *, summary_delay=0.15, answer_delay=0.0):
        self.summary_delay = summary_delay
        self.answer_delay = answer_delay
        self.calls = []
        self.cancelled = []
        self.finished = []

    def context_window(self):
        return 40_000

    async def complete(self, messages, tools=None, **kwargs):
        summary = "Your task is to create a detailed summary" in messages[-1]["content"]
        self.calls.append((summary, threading.get_ident()))
        try:
            await asyncio.sleep(self.summary_delay if summary else self.answer_delay)
        except asyncio.CancelledError:
            self.cancelled.append(summary)
            raise
        self.finished.append(summary)
        return llm_response(content="<summary>Earlier context.</summary>" if summary else "answer")


@pytest.mark.parametrize("long", [False, True])
def test_default_summary_obeys_the_generation_ceiling_and_yields_to_observers(long):
    model = SlowModel(answer_delay=0.15 if not long else 0)
    session = build_session(agent=Agent(name="timing", system_prompt="system"), llm=model, llm_timeout=0.02)
    session.messages = history(long)

    async def scenario():
        started = time.monotonic()
        observed = []

        async def observer():
            await asyncio.sleep(0.01)
            observed.append(time.monotonic() - started)

        observation = asyncio.create_task(observer())
        await asyncio.sleep(0)
        with pytest.raises(GenerationTimeoutError):
            await asyncio.wait_for(session.run_loop(), 0.3)
        elapsed = time.monotonic() - started
        await observation
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)
        assert elapsed < 0.10
        assert observed[0] < 0.10

    asyncio.run(scenario())
    assert [call[0] for call in model.calls] == [long]
    assert model.cancelled == [long]
    assert model.finished == []
    assert session.pending_cleanup_tasks == ()


@pytest.mark.parametrize("long", [False, True])
def test_outer_cancellation_reaches_the_owned_summary_and_prevents_an_answer(long):
    model = SlowModel(answer_delay=0.15 if not long else 0)
    session = build_session(agent=Agent(name="cancel", system_prompt="system"), llm=model, llm_timeout=None)
    session.messages = history(long)

    async def scenario():
        task = asyncio.create_task(session.run_loop())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.3)
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)

    asyncio.run(scenario())
    assert [call[0] for call in model.calls] == [long]
    assert model.calls[0][1] == threading.get_ident()
    assert model.cancelled == [long]
    assert model.finished == []
    assert session.pending_cleanup_tasks == ()
