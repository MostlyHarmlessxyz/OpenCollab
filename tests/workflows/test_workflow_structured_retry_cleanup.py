"""A structured corrective session retains the run's provider capacity."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.types import LLMResponse, Usage


@pytest.mark.parametrize("draining", [False, True])
async def test_structured_retry_requires_the_prior_model_to_finish(tmp_path, monkeypatch, draining):
    monkeypatch.delenv("OPENCOLLAB_UNBOUNDED_LIMITS", raising=False)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    release = asyncio.Event()
    cancelled = asyncio.Event()
    active = peak = calls = 0

    async def complete(llm, messages, **kwargs):
        nonlocal active, peak, calls
        active += 1
        peak = max(peak, active)
        calls += 1
        try:
            if calls == 1:
                if not draining:
                    raise RuntimeError("completed provider failure")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    return LLMResponse(content="late response", usage=Usage(5, 3), finish_reason="stop")
            return LLMResponse(tool_calls=[{
                "id": "result", "type": "function",
                "function": {"name": "structured_output", "arguments": json.dumps({"answer": "ready"})},
            }], usage=Usage(5, 3), finish_reason="tool_calls")
        finally:
            active -= 1

    monkeypatch.setattr(LLMClient, "complete", complete)

    async def flow(ctx, _args):
        try:
            return await ctx.agent("Produce an answer", schema={
                "type": "object", "properties": {"answer": {"type": "string"}},
                "required": ["answer"], "additionalProperties": False,
            })
        finally:
            release.set()

    client = OpenCollab(tmp_path, model="test-model", provider="openai", api_key="test-key",  # pragma: allowlist secret
                        config={"llm_timeout": 0.05, "llm_max_retries": 0})
    result = await asyncio.wait_for(client.workflow(flow, budget=100_000, concurrency=1, trace=False), timeout=5)

    assert peak == 1
    assert active == 0
    assert calls == (1 if draining else 2)
    assert cancelled.is_set() is draining
    assert result.output == (None if draining else {"answer": "ready"})
    assert result.tokens == 8
