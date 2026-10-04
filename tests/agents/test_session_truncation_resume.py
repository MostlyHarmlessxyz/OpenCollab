"""Provider truncation keeps the next public session turn usable."""

import pytest

from opencollab.application.session import Session, SessionRuntime
from opencollab.domain.session import SessionPhase
from tests.support.session_run_loop_test_support import (
    FakeLLM,
    FakeTracer,
    build_runner,
    llm_response,
    tool_call,
)


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "model_context_window_exceeded"])
@pytest.mark.parametrize("content", [None, "partial answer"])
@pytest.mark.parametrize("with_tool_call", [False, True])
async def test_truncated_turn_preserves_text_and_trace_without_replaying_tools(
    finish_reason, content, with_tool_call
):
    call = tool_call(arguments='{"value": 1}')
    response = llm_response(
        content=content,
        tool_calls=[call] if with_tool_call else [],
        finish_reason=finish_reason,
        provider_state={"anthropic_content": [{"type": "tool_use", "id": call["id"]}]}
        if with_tool_call else None,
    )
    response.provider_items = [{"type": "function_call", "call_id": call["id"]}] if with_tool_call else []
    model = FakeLLM([response, llm_response(content="next answer")])
    tracer = FakeTracer()
    runner = build_runner(llm=model, tracer=tracer)
    runner.tool_execution.environment = None
    runtime = SessionRuntime(
        state=runner.state,
        event_bus=runner.event_publisher,
        llm=model,
        store=None,
        tool_execution=runner.tool_execution,
        runner=runner,
        auto_save_path=None,
    )
    session = Session(runner.agent, runtime=runtime)

    await session.add_user_message("first request")
    assert await session.run_loop() == (content or "")
    assert session.phase is SessionPhase.STOPPED
    assert session.state.terminal_reason == "output truncated: provider reached its generation limit"
    assert runner.tool_execution.calls == []

    await session.add_user_message("continue")
    assert await session.run_loop() == "next answer"
    assert session.phase is SessionPhase.DONE
    assert len(model.calls) == 2
    next_prompt = model.calls[1]["messages"]
    assert not any(message.get("tool_calls") for message in next_prompt)
    assert not any(message.get("response_items") or message.get("provider_state") for message in next_prompt)
    partial_replies = [message for message in next_prompt if message["role"] == "assistant"]
    assert partial_replies == ([{"role": "assistant", "content": content}] if content else [])
    assert runner.tool_execution.calls == []
    assert session.used_tokens == 10

    first_trace = next(step["payload"] for step in tracer.steps if step["step_type"] == "llm_call")
    assert first_trace["finish_reason"] == finish_reason
    assert first_trace["tool_calls"] == (
        [{"id": call["id"], "name": "fake_tool", "arguments": '{"value": 1}'}]
        if with_tool_call else None
    )
