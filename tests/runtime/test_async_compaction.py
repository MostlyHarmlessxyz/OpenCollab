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


def test_forced_compaction_retry_awaits_summary_and_restores_forced_flags():
    class OverflowModel(SlowModel):
        async def complete(self, messages, tools=None, **kwargs):
            if not self.calls:
                self.calls.append((False, threading.get_ident()))
                raise ValueError("fixture context overflow")
            return await super().complete(messages, tools, **kwargs)

    model = OverflowModel()
    session = build_session(agent=Agent(name="overflow", system_prompt="system"), llm=model, llm_timeout=0.02)
    session.messages = history(False)
    session.runner._is_context_overflow = lambda error: isinstance(error, ValueError)

    async def scenario():
        with pytest.raises(GenerationTimeoutError):
            await asyncio.wait_for(session.run_loop(), 0.3)
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)

    asyncio.run(scenario())
    assert [call[0] for call in model.calls] == [False, True]
    assert model.cancelled == [True]
    assert all(not getattr(shaper, "_forced", False) for shaper in session.runner.shaper._shapers)


def test_owned_summary_client_closes_after_cancellation(monkeypatch):
    from opencollab.bootstrap import container

    clients = []

    class OwnedModel(SlowModel):
        def __init__(self, **kwargs):
            super().__init__()
            self.close_calls = 0
            clients.append(self)

        async def close(self):
            self.close_calls += 1

    monkeypatch.setattr(container, "LLMClient", OwnedModel)
    session = build_session(agent=Agent(name="owned", system_prompt="system"), llm_timeout=0.02)
    session.messages = history()

    async def scenario():
        with pytest.raises(GenerationTimeoutError):
            await asyncio.wait_for(session.run_loop(), 0.3)
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)
        await session.aclose()

    asyncio.run(scenario())
    assert len(clients) == 2
    assert clients[0].calls == []
    assert clients[1].cancelled == [True]
    assert [client.close_calls for client in clients] == [1, 1]
