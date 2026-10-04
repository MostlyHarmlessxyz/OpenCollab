"""Visible Chat refusal text through the real SDK and public agent runtime."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient

_REFUSAL = "I cannot provide that response. I can explain the general principles."
_ANSWER = "Here are the general principles."
_USAGE = {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}


def _reply(stream, content, refusal, finish_reason):
    common = {"id": "chatcmpl-refusal", "created": 1, "model": "gpt-4o"}
    if not stream:
        return httpx.Response(200, json={
            **common,
            "object": "chat.completion",
            "usage": _USAGE,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish_reason,
            }],
        })

    common["object"] = "chat.completion.chunk"
    deltas = [{"role": "assistant", "content": content if not content else "", "refusal": None}]
    for field, text in (("content", content), ("refusal", refusal)):
        if text:
            deltas.extend({field: text[offset:offset + 7]} for offset in range(0, len(text), 7))
    frames = [
        {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        for delta in deltas
    ]
    frames.extend([
        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
        {**common, "choices": [], "usage": _USAGE},
    ])
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body.encode())


async def _run_reply(tmp_path, stream, content, refusal, finish_reason="stop"):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return _reply(stream, content, refusal, finish_reason)

    client = LLMClient(model="gpt-4o", api_key="unused", max_retries=0, stream_chat=stream)  # pragma: allowlist secret
    await client._openai.close()
    client._openai = openai.AsyncOpenAI(
        api_key="unused",  # pragma: allowlist secret
        base_url="https://refusal.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    artifacts = tmp_path / "artifacts"
    try:
        result = await OpenCollab(
            tmp_path, model="gpt-4o", provider="openai", api_key="unused",  # pragma: allowlist secret
            config={"thinking": False, "top_p": None},
        ).agent(
            "Return a brief visible response.", llm=client, tools=(),
            max_steps=3, budget=10_000, timeout=10, cleanup_timeout=1,
            trace=False, artifacts=artifacts,
        )
        history = json.loads((artifacts / "agent.json").read_text())["messages"]
        return result, requests, history
    finally:
        await client.close()


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("content", [None, "", " \t\n"], ids=["absent", "empty", "whitespace"])
async def test_refusal_only_reply_is_returned_and_persisted_once(tmp_path, stream, content):
    result, requests, history = await _run_reply(tmp_path, stream, content, _REFUSAL)

    assert result.output == _REFUSAL
    assert result.status == "completed"
    assert result.tokens == 20
    assert len(requests) == 1
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == _REFUSAL


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("refusal", [None, _REFUSAL], ids=["content", "mixed"])
async def test_visible_content_keeps_precedence_over_refusal(tmp_path, stream, refusal):
    result, requests, history = await _run_reply(tmp_path, stream, _ANSWER, refusal)

    assert result.output == _ANSWER
    assert result.tokens == 20
    assert len(requests) == 1
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == _ANSWER


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_empty_content_filter_reply_completes_without_empty_stop_retry(tmp_path, stream):
    result, requests, _history = await _run_reply(tmp_path, stream, None, None, "content_filter")

    assert result.output == ""
    assert result.status == "completed"
    assert result.tokens == 20
    assert len(requests) == 1
