"""Every trajectory record says which agent wrote it.

A team writes one trajectory for all of its agents. ``aid`` used to appear
only inside the payloads that happened to include it, and ``role`` in fewer,
so a record such as a refused tool call could not be attributed to anyone.
Each record now carries ``aid`` and ``role`` beside ``run_id``; records the
scheduler writes on no agent's behalf carry ``null`` for both.
"""

from __future__ import annotations

import json

from opencollab.adapters.trace import Tracer
from opencollab.application.event_bus import EventBus
from opencollab.bootstrap import build_session as Session
from tests.support.session_characterization_test_support import (
    FakeAgent,
    FakeLLMClient,
    FakeTool,
    llm_response,
    run,
    tool_call,
)


def _records(tracer: Tracer) -> list[dict]:
    tracer.flush()
    with open(tracer._path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_every_record_a_session_writes_names_its_agent(tmp_path) -> None:
    tracer = Tracer("run-1", output_dir=str(tmp_path), filename="trajectory.jsonl")
    agent = FakeAgent(tools=[FakeTool(name="known_tool")])
    agent.name = "coder"
    fake_llm = FakeLLMClient(
        [
            llm_response(
                tool_calls=[tool_call(name="missing_tool", arguments="{}")],
                finish_reason="tool_calls",
            ),
            llm_response(content="recovered"),
        ]
    )
    session = Session(agent=agent, llm=fake_llm, tracer=tracer, event_sink=EventBus(None), aid=3)
    # The public tracer is the agent's own view, so handing it back through the
    # setter (which re-wires the runner and the tool executor) keeps attribution.
    session.tracer = session.tracer
    try:
        assert run(session.run_loop()) == "recovered"
        records = _records(tracer)
    finally:
        tracer.close()

    assert records
    assert {(r["aid"], r["role"]) for r in records} == {(3, "coder")}
    refused = [r for r in records if r["type"] == "tool_error"]
    assert refused and refused[0]["run_id"] == "run-1"


def test_a_record_written_on_no_agents_behalf_says_so(tmp_path) -> None:
    tracer = Tracer("run-1", output_dir=str(tmp_path), filename="trajectory.jsonl")
    try:
        tracer.log_step("assigned.topology_edges", {"edges": []})
        (record,) = _records(tracer)
    finally:
        tracer.close()
    assert record["aid"] is None
    assert record["role"] is None


def test_the_role_is_read_when_the_record_is_written(tmp_path) -> None:
    """A prebuilt teammate is named after its session is built."""
    tracer = Tracer("run-1", output_dir=str(tmp_path), filename="trajectory.jsonl")
    agent = FakeAgent()
    agent.name = "default"
    view = tracer.for_agent(2, agent)
    agent.name = "tester"
    try:
        view.log_step("tool_exec", {})
        (record,) = _records(tracer)
    finally:
        tracer.close()
    assert (record["aid"], record["role"]) == (2, "tester")
