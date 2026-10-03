"""Role declarations compose with context policy and per-run identity."""

from __future__ import annotations

import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.bootstrap import container


@pytest.mark.parametrize("declared_budget", [False, True], ids=["shared", "per-role"])
async def test_team_budget_and_models_preserve_context_and_run_identity(
    tmp_path, monkeypatch, declared_budget,
):
    calls = []

    class OfflineLLM:
        def __init__(self, *, model, **kwargs):
            self.model = model
            self.turn = 0

        def context_window(self):
            return 200_000

        async def close(self):
            pass

        async def complete(self, messages, **kwargs):
            self.turn += 1
            calls.append(self.model)
            usage = Usage(input_tokens=10, output_tokens=5)
            if self.model == "lead-model" and self.turn == 1:
                return LLMResponse(
                    tool_calls=[{
                        "id": "handoff", "type": "function",
                        "function": {
                            "name": "message_agent",
                            "arguments": json.dumps({
                                "to_aid": 1, "summary": "check", "content": "Reply when ready.",
                            }),
                        },
                    }],
                    usage=usage, finish_reason="tool_calls",
                )
            return LLMResponse(content="finished", usage=usage, finish_reason="stop")

    monkeypatch.setattr(container, "LLMClient", OfflineLLM)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    team_file = tmp_path / "team.yaml"
    default_budget = "budget: {tokens: 30000}\n" if declared_budget else ""
    lead_budget = "    budget: {tokens: 45000}\n" if declared_budget else ""
    team_file.write_text(
        "entry: lead\ncontext: no_history_compaction\n" + default_budget
        + "roles:\n  lead:\n    prompt: Lead the handoff.\n"
        + "    model: lead-model\n    tools: [message_agent]\n" + lead_budget
        + "  coder:\n    prompt: Reply to the lead.\n    tools: []\n"
        + "topology:\n  lead: [coder]\n  coder: []\n",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    client = OpenCollab(workspace, model="fallback-model", config={"budget": 90_000})
    result = await client.team(
        "Ask the coder to check the handoff.", config=team_file,
        artifacts=artifacts, prebuild_team=True, serialize_turns=True,
        use_worktrees=False, max_steps=8, timeout=5,
    )

    assert result.status == "completed", result.reason
    assert {"lead-model", "fallback-model"} <= set(calls)
    records = [json.loads(line) for line in (artifacts / "trajectory.jsonl").read_text().splitlines()]
    manifest = json.loads((artifacts / "team.json").read_text())
    run_id = result.metrics["run_id"]
    assert run_id.startswith("team-")
    assert {record["run_id"] for record in records} == {run_id}
    assert manifest["run_id"] == run_id
    declaration = next(record["payload"] for record in records if record["type"] == "assigned.topology_nodes")
    nodes = {node["role"]: node for node in declaration["nodes"]}
    assert {role: node["model"] for role, node in nodes.items()} == {
        "lead": "lead-model", "coder": "fallback-model",
    }
    assert declaration["budget_source"] == ("team_file" if declared_budget else "shared_rule")
    if declared_budget:
        assert {role: node["token_allowance"] for role, node in nodes.items()} == {
            "lead": 45_000, "coder": 30_000,
        }
    else:
        assert nodes["lead"]["token_allowance"] == nodes["coder"]["token_allowance"] > 0
    policies = [record["payload"] for record in records if record["type"] == "session.history_compaction"]
    assert len(policies) >= 2
    assert all(policy["context_policy"] == "no_history_compaction" for policy in policies)
