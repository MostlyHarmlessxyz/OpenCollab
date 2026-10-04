"""Provider diagnostics honor configured wire protocol and own SDK resources."""

import json
import os
from unittest.mock import patch

import httpx
import openai
import pytest

from scripts.check_dashscope import request_completion


@pytest.mark.parametrize("fail", [False, True])
async def test_diagnostic_responses_configuration_and_sdk_cleanup(monkeypatch, tmp_path, fail):
    requests = []
    transports = []

    async def handler(request):
        requests.append(request)
        if request.url.path != "/v1/responses":
            return httpx.Response(404, json={"error": {"message": "Only Responses is supported"}})
        if fail:
            return httpx.Response(401, json={"error": {"message": "Invalid test credentials"}})
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "resp_diagnostic", "object": "response", "created_at": 1,
            "status": "completed", "model": body["model"], "instructions": body.get("instructions"),
            "error": None, "incomplete_details": None, "usage": None,
            "output": [{"id": "msg_diagnostic", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "connected", "annotations": []}]}],
        })

    sdk_type = openai.AsyncOpenAI

    def create_sdk(**kwargs):
        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transports.append(transport)
        return sdk_type(**kwargs, http_client=transport)

    monkeypatch.setattr(openai, "AsyncOpenAI", create_sdk)
    config_path = tmp_path / "config.env"
    config_path.write_text(
        "OPENCOLLAB_MODEL=o1-pro\nOPENCOLLAB_PROVIDER=openai\nOPENCOLLAB_WIRE_PROTOCOL=responses\n"
        "OPENCOLLAB_API_KEY=test-placeholder\nOPENCOLLAB_BASE_URL=https://provider.invalid/v1\n"  # pragma: allowlist secret
        "OPENCOLLAB_LLM_MAX_RETRIES=0\n",
    )
    with patch.dict(os.environ, {"OPENCOLLAB_CONFIG_FILE": str(config_path)}, clear=True):
        if fail:
            with pytest.raises(openai.AuthenticationError):
                await request_completion("connect", workspace=tmp_path)
        else:
            assert await request_completion("connect", workspace=tmp_path) == "connected"

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    assert json.loads(requests[0].content)["stream"] is False
    assert len(transports) == 1
    assert transports[0].is_closed
