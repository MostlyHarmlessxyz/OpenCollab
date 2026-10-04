"""Anthropic SDK requests preserve separate connection and request timeouts."""

from __future__ import annotations

import asyncio

import anthropic
import httpx
import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport


@pytest.fixture(autouse=True)
def isolate_usage_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))


@pytest.mark.parametrize("request_timeout,connect_timeout", [
    (0.2, 0.02), (0.2, 0.5), (None, 0.02), (0.2, None), (None, None), (600.0, 30.0),
])
async def test_real_anthropic_requests_keep_all_timeout_components(monkeypatch, request_timeout, connect_timeout):
    requests = []

    async def handler(request):
        requests.append(request.extensions["timeout"])
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    async with LLMClient(
        provider="anthropic", model="claude-sonnet-4-6",
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", max_retries=0,
        request_timeout=request_timeout, connect_timeout=connect_timeout,
    ) as client:
        assert (await client.complete([{"role": "user", "content": "Return ok"}])).content == "ok"

    assert requests == [{
        "connect": connect_timeout, "read": request_timeout, "write": request_timeout, "pool": request_timeout,
    }]


@pytest.mark.parametrize("connect_timeout,timed_out", [(0.02, True), (0.2, False)])
async def test_public_model_client_enforces_the_configured_connection_allowance(
    monkeypatch, tmp_path, connect_timeout, timed_out,
):
    requests = []
    connect_delay = 0.06

    async def handler(request):
        timeout = request.extensions["timeout"]
        requests.append(timeout)
        allowance = timeout["connect"]
        await asyncio.sleep(min(connect_delay, allowance))
        if allowance < connect_delay:
            raise httpx.ConnectTimeout("Controlled connection exceeded its allowance", request=request)
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "anthropic", handler)
    app = OpenCollab(
        tmp_path, provider="anthropic", model="claude-sonnet-4-6",
        api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"llm_timeout": 0.2, "llm_connect_timeout": connect_timeout, "llm_max_retries": 0},
    )
    async with app.create_model_client() as client:
        if timed_out:
            with pytest.raises(anthropic.APITimeoutError):
                await client.complete([{"role": "user", "content": "Return ok"}])
        else:
            assert (await client.complete([{"role": "user", "content": "Return ok"}])).content == "ok"

    assert requests == [{"connect": connect_timeout, "read": 0.2, "write": 0.2, "pool": 0.2}]
