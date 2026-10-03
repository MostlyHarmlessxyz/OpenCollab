"""Compaction calls share session budget, usage and provider traces."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.adapters.trace import Tracer
from opencollab.bootstrap import build_session
from opencollab.bootstrap.context_policy import ContextPolicy
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from opencollab.domain.token_estimation import estimate_request_tokens
from tests.runtime.test_async_compaction import history


class AccountingModel:
    def __init__(self, summary_content="<summary>Preserved context.</summary>", summary_tokens=7_600):
        self.summary_content = summary_content
        self.summary_tokens = summary_tokens
        self.calls = []

    def context_window(self):
        return 40_000

    async def complete(self, messages, tools=None, **kwargs):
        summary = "Your task is to create a detailed summary" in messages[-1]["content"]
        self.calls.append({"summary": summary, "messages": copy.deepcopy(messages), "kwargs": kwargs})
        return LLMResponse(
            content=self.summary_content if summary else "answer",
            finish_reason="stop",
            usage=Usage(input_tokens=self.summary_tokens - 2 if summary else 8, output_tokens=2,
                        cache_read_tokens=20 if summary else 0, reasoning_tokens=1 if summary else None,
                        raw_usage={"provider_input": self.summary_tokens - 2} if summary else {}),
            reasoning="provider reasoning" if summary else None,
            provider_model="fixture-provider-model",
            transport_timing={"first_token_seconds": 0.001},
        )


@pytest.fixture(autouse=True)
def disable_steering(monkeypatch):
    monkeypatch.setenv("OPENCOLLAB_BUDGET_NUDGE_MODE", "off")
    monkeypatch.setenv("OPENCOLLAB_WRITE_NUDGE_MODE", "off")


@pytest.mark.parametrize("budget", [30_000, None])
@pytest.mark.parametrize("content", ["<summary>Preserved context.</summary>", "empty parsed summary"])
def test_summary_usage_and_native_trace_fields_are_recorded(tmp_path, budget, content):
    model = AccountingModel(content)
    tracer = Tracer(run_id="summary-usage", output_dir=str(tmp_path))
    session = build_session(agent=Agent(name="accounting", system_prompt="system"), llm=model,
                            tracer=tracer, max_budget_tokens=budget)
    session.messages = history()
    try:
        assert asyncio.run(session.run_loop()) == "answer"
        tracer.flush()
    finally:
        tracer.close()
    assert session.used_tokens == 7_610
    assert [call["summary"] for call in model.calls] == [True, False]
    rows = [json.loads(line) for line in open(tracer.path, encoding="utf-8")]
    calls = [row for row in rows if row["type"] == "llm_call"]
    assert len(calls) == 2
    summary = next(row for row in calls if row["payload"].get("purpose") == "summary")
    assert summary["tokens"] == 7_600
    assert summary["payload"]["usage"]["input_tokens"] == 7_598
    assert summary["payload"]["usage"]["cache_read_tokens"] == 20
    assert summary["payload"]["usage"]["raw_usage"] == {"provider_input": 7_598}
    assert summary["payload"]["usage"]["reasoning_tokens"] == 1
    assert summary["payload"]["reasoning"] == "provider reasoning"
    assert summary["payload"]["provider_model"] == "fixture-provider-model"
    assert summary["payload"]["transport_timing"] == {"first_token_seconds": 0.001}
    assert summary["payload"]["request_tool_names"] == []
    assert summary["payload"]["request_tool_choice"] is None
    terminal = next(row for row in rows if row["type"] == "session_terminal")
    assert terminal["payload"]["used_tokens"] == 7_610


@pytest.mark.parametrize("long", [False, True])
def test_remaining_budget_stops_before_any_model_request(long):
    model = AccountingModel()
    session = build_session(agent=Agent(name="low-budget", system_prompt="system"), llm=model,
                            max_budget_tokens=5_000)
    session.messages = history(long)
    session.used_tokens = 4_900
    asyncio.run(session.run_loop())
    assert model.calls == []
    assert session.used_tokens == 4_900
    assert session.phase is SessionPhase.STOPPED
    assert "budget exhausted before model call" in session.state.terminal_reason


def test_summary_reserves_input_and_limits_output():
    model = AccountingModel(summary_tokens=20)
    session = build_session(agent=Agent(name="bounded", system_prompt="system", max_tokens_per_step=50_000),
                            llm=model, max_budget_tokens=10_000)
    session.messages = history()
    asyncio.run(session.run_loop())
    summary, answer = model.calls
    assert summary["summary"] is True
    assert summary["kwargs"]["max_output_tokens"] == 10_000 - estimate_request_tokens(summary["messages"])
    assert answer["kwargs"]["max_output_tokens"] == 9_980 - estimate_request_tokens(answer["messages"])


def test_successful_summary_cache_reuse_charges_once():
    model = AccountingModel()
    session = build_session(agent=Agent(name="cached", system_prompt="system"), llm=model)
    session.messages = history()

    async def scenario():
        first = await session.runner._shape_and_trace(session.messages)
        second = await session.runner._shape_and_trace(session.messages)
        assert first == second
        assert session.used_tokens == 7_600
        assert await session.run_loop() == "answer"

    asyncio.run(scenario())
    assert [call["summary"] for call in model.calls] == [True, False]
    assert session.used_tokens == 7_610


def test_summary_overspend_stops_before_an_answer_request():
    model = AccountingModel(summary_tokens=40_000)
    session = build_session(agent=Agent(name="overspend", system_prompt="system"), llm=model,
                            max_budget_tokens=30_000)
    session.messages = history()
    asyncio.run(session.run_loop())
    assert [call["summary"] for call in model.calls] == [True]
    assert session.used_tokens == 40_000
    assert session.phase is SessionPhase.STOPPED


def test_disabled_history_compaction_has_only_answer_usage():
    model = AccountingModel()
    session = build_session(agent=Agent(name="disabled", system_prompt="system"), llm=model,
                            context_policy=ContextPolicy(name="no_history_compaction"))
    session.messages = history()
    asyncio.run(session.run_loop())
    assert [call["summary"] for call in model.calls] == [False]
    assert session.used_tokens == 10
