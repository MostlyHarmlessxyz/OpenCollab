"""Session wind-down preserves a reusable Agent configuration template."""

from __future__ import annotations

import asyncio

import pytest

from opencollab.bootstrap import build_session
from tests.support.session_run_test_support import FakeLLM, agent_with_submit, llm_response


@pytest.mark.parametrize("create_second_before_first_turn", [False, True])
def test_wind_down_preserves_tools_for_a_session_reusing_the_template(
    create_second_before_first_turn,
):
    template = agent_with_submit()
    first_model = FakeLLM([llm_response(content="first protected answer")])
    second_model = FakeLLM([llm_response(content="second answer")])
    first = build_session(agent=template, llm=first_model, max_budget_tokens=50_000)

    async def scenario():
        second = None
        if create_second_before_first_turn:
            second = build_session(agent=template, llm=second_model, max_budget_tokens=50_000)
            await second.add_user_message("Read for an independent turn.")
        await first.add_user_message("Inspect and submit.")
        first.runner.configure_enforcement(enforcement_strength="needs-enforcement")
        first.used_tokens = 25_000
        await first.run_loop()
        if second is None:
            second = build_session(agent=template, llm=second_model, max_budget_tokens=50_000)
            await second.add_user_message("Read for a later turn.")
        await second.run_loop()
        return second

    second = asyncio.run(scenario())

    assert first.state.wind_down_done is True
    assert [tool.name for tool in first.agent.tools] == ["submit_findings"]
    assert [spec["function"]["name"] for spec in second_model.calls[0]["tools"]] == [
        "file_read", "submit_findings"
    ]
    assert second_model.calls[0]["tool_choice"] is None
    assert [tool.name for tool in template.tools] == ["file_read", "submit_findings"]
    assert template.tool_choice is None
    assert first.agent is first.runner.agent is first.tool_execution.agent
    assert second.agent is second.runner.agent is second.tool_execution.agent
    assert first.agent is not second.agent
    assert first.agent.tools[0] is template.tools[1]


def test_separate_templates_keep_their_tool_configuration():
    first_template = agent_with_submit()
    second_template = agent_with_submit()
    first = build_session(agent=first_template, llm=FakeLLM([llm_response(content="first")]))
    second = build_session(agent=second_template, llm=FakeLLM([llm_response(content="second")]))

    async def scenario():
        await first.add_user_message("first")
        await first.run_loop()
        await second.add_user_message("second")
        await second.run_loop()

    asyncio.run(scenario())
    assert [tool.name for tool in first.agent.tools] == ["file_read", "submit_findings"]
    assert [tool.name for tool in second.agent.tools] == ["file_read", "submit_findings"]
