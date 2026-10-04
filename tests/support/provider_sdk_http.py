"""Real provider SDKs with in-memory HTTP responses for session tests."""

from __future__ import annotations

import json

import anthropic
import httpx
import openai

from tests.support.responses_provider_test_support import message_item


def install_sdk_transport(monkeypatch, provider, handler):
    package = anthropic if provider == "anthropic" else openai
    name = "AsyncAnthropic" if provider == "anthropic" else "AsyncOpenAI"
    sdk_type = getattr(package, name)

    def create_sdk(**kwargs):
        return sdk_type(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(package, name, create_sdk)


def completion_http_response(request, *, output_tokens=2, text="ok"):
    body = json.loads(request.content)
    model = body["model"]
    if request.url.path.endswith("/messages"):
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": output_tokens},
        })
    if request.url.path.endswith("/responses"):
        item = message_item(text)
        response = {
            "id": "resp_test", "object": "response", "status": "completed", "model": model,
            "output": [item], "error": None, "incomplete_details": None,
            "usage": {"input_tokens": 2, "output_tokens": output_tokens, "total_tokens": 2 + output_tokens},
        }
        events = [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        text = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)
    return httpx.Response(200, json={
        "id": "chatcmpl_test", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": output_tokens, "total_tokens": 2 + output_tokens},
    })
