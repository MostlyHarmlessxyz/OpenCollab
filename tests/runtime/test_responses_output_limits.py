"""Provider output limits preserve terminal status and reported consumption."""

import pytest

from opencollab.adapters.llm.responses_provider import _consume_stream, _parse_stream
from tests.support.responses_provider_test_support import FakeStream, completed_response, function_item, ns


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.asyncio
async def test_output_limit_keeps_usage_without_executing_partial_tool(streamed):
    from opencollab.adapters.llm.responses_provider import parse_responses_response
    from opencollab.bootstrap import build_session
    from opencollab.domain.session import SessionPhase
    from tests.support.session_characterization_test_support import FakeAgent, FakeLLMClient

    item = {**function_item("partial", "file_write", '{"path":'), "status": "incomplete"}
    response = completed_response(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[item],
    )
    messages = [{"role": "user", "content": "write the file"}]
    if streamed:
        state = await _consume_stream(FakeStream([
            ns(type="response.function_call_arguments.delta", output_index=0, delta=item["arguments"]),
            ns(type="response.incomplete", response=response),
        ]), 1, 1)
        parsed = _parse_stream(state, messages, "gpt-fake")
    else:
        parsed = parse_responses_response(response, messages, expected_model="gpt-fake")
    assert parsed.finish_reason == "max_tokens"
    assert parsed.tool_calls == []
    assert parsed.provider_items == []
    assert parsed.usage.total_tokens == 150
    assert parsed.usage.estimated is False
    session = build_session(agent=FakeAgent(), llm=FakeLLMClient([parsed]))
    await session.run_loop()
    assert session.phase is SessionPhase.STOPPED
    assert session.used_tokens == 150
    assert not any(message.get("tool_calls") for message in session.messages)
