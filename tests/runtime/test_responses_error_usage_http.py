"""Real SDK and Session accounting for reported Responses failure costs."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from opencollab.adapters.llm import client as client_module
from opencollab.adapters.llm import retry
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.responses_errors import ResponsesProtocolError
from opencollab.adapters.storage import SessionStore
from opencollab.application.session_run import GenerationTimeoutError
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionPhase
from tests.runtime.test_async_compaction import history
from tests.support.provider_sdk_http import install_sdk_transport
from tests.support.responses_provider_test_support import message_item

KNOWN_USAGE = {
    "input_tokens": 30, "output_tokens": 7, "total_tokens": 37,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _response(body, spec):
    kind = spec.get("kind", "answer")
    items = [] if kind in {"filter", "empty", "failed"} else [message_item(spec.get("text", "ok"))]
    return {
        "id": "resp_usage_test", "object": "response", "created_at": 1, "model": body["model"],
        "instructions": body.get("instructions"),
        "status": "incomplete" if kind == "filter" else "failed" if kind == "failed" else "completed",
        "error": (
            {"code": "server_error", "message": "An internal server error occurred."} if kind == "failed" else None
        ),
        "incomplete_details": {"reason": "content_filter"} if kind == "filter" else None,
        "output": items, "usage": spec.get("usage", KNOWN_USAGE),
    }


def _payload(body, spec):
    response = _response(body, spec)
    if not body["stream"]:
        return "application/json", json.dumps(response).encode()
    events = [{"type": "response.output_item.done", "sequence_number": i,
               "output_index": i, "item": item} for i, item in enumerate(response["output"])]
    terminal = {"completed": "response.completed", "failed": "response.failed", "incomplete": "response.incomplete"}
    events.append({"type": terminal[response["status"]], "sequence_number": len(events), "response": response})
    return "text/event-stream", "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


@pytest.fixture
def local_responses_server():
    servers = []

    def start(specs):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                spec = specs[min(len(requests) - 1, len(specs) - 1)]
                content_type, payload = _payload(body, spec)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/v1", requests

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def usage_log(monkeypatch, tmp_path):
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(path))
    monkeypatch.delenv("OPENCOLLAB_EXTERNAL_PROVIDER_ISOLATION", raising=False)
    monkeypatch.setattr(retry, "extract_retry_after_seconds", lambda _error: 0.0)
    monkeypatch.setattr(retry.random, "uniform", lambda _lo, _hi: 0.0)
    return path


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _client(model, base_url, *, retries=0):
    return LLMClient(
        model=model, wire_protocol="responses", api_key="controlled-local",  # pragma: allowlist secret
        base_url=base_url, max_retries=retries, context_window=40000,
        first_event_timeout=None, request_timeout=None,
    )


def _agent(model="gpt-4o"):
    return Agent(name="reader", system_prompt="Reply briefly", model=model, wire_protocol="responses")


async def test_ordinary_success_usage_is_charged_once(local_responses_server, usage_log):
    base_url, requests = local_responses_server([{}])
    async with _client("gpt-4o", base_url) as client:
        session = build_session(agent=_agent(), llm=client)
        await session.add_user_message("Give a short answer")
        assert await session.run_loop() == "ok"
    assert session.phase is SessionPhase.DONE
    assert session.used_tokens == 37
    assert session.state.context_tokens == 30
    assert len(requests) == 1
    records = _records(usage_log)
    assert len(records) == 1
    assert records[0]["usage"]["total_tokens"] == 37
    assert records[0]["usage"]["raw_usage"] == KNOWN_USAGE


@pytest.mark.parametrize("model", ["gpt-4o", "o1-pro"])
@pytest.mark.parametrize("kind", ["filter", "failed"])
async def test_reported_terminal_error_usage_reaches_session_and_ledger(model, kind, local_responses_server, usage_log):
    base_url, requests = local_responses_server([{"kind": kind}])
    async with _client(model, base_url) as client:
        session = build_session(agent=_agent(model), llm=client)
        await session.add_user_message("Give a short answer")
        with pytest.raises(ResponsesProtocolError) as captured:
            await session.run_loop()
    assert session.phase is SessionPhase.ERROR
    assert session.used_tokens == 37
    assert len(requests) == 1
    assert requests[0]["stream"] is (model == "gpt-4o")
    if kind == "failed":
        assert captured.value.code == "server_error"
    else:
        assert "content_filter" in str(captured.value)
    assert captured.value.usage.raw_usage == KNOWN_USAGE
    records = _records(usage_log)
    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["usage"]["total_tokens"] == 37
    assert records[0]["usage"]["estimated"] is False


@pytest.mark.parametrize("model", ["gpt-4o", "o1-pro"])
@pytest.mark.parametrize("first_kind", ["empty", "failed"])
@pytest.mark.parametrize("recover", [False, True])
async def test_retry_usage_is_charged_once_with_final_request_context(
    model, first_kind, recover, local_responses_server, usage_log,
):
    specs = [{"kind": first_kind}, {"kind": "answer" if recover else first_kind}]
    base_url, requests = local_responses_server(specs)
    async with _client(model, base_url, retries=1) as client:
        session = build_session(agent=_agent(model), llm=client)
        await session.add_user_message("Give a short answer")
        if recover:
            assert await session.run_loop() == "ok"
            assert session.phase is SessionPhase.DONE
            assert session.state.context_tokens == 30
        else:
            with pytest.raises(ResponsesProtocolError):
                await session.run_loop()
            assert session.phase is SessionPhase.ERROR
    assert len(requests) == 2
    assert session.used_tokens == 74
    records = _records(usage_log)
    assert len(records) == 1
    assert records[0]["status"] == ("success" if recover else "error")
    assert records[0]["usage"]["input_tokens"] == 60
    assert records[0]["usage"]["output_tokens"] == 14
    assert records[0]["usage"]["total_tokens"] == 74
    assert records[0]["usage"]["raw_usage"]["attempts"] == [KNOWN_USAGE, KNOWN_USAGE]


async def test_retry_cost_can_exhaust_session_budget(local_responses_server, usage_log):
    usage = {**KNOWN_USAGE, "input_tokens": 600, "output_tokens": 300, "total_tokens": 900}
    base_url, requests = local_responses_server([{"kind": "empty", "usage": usage}, {"usage": usage}])
    async with _client("gpt-4o", base_url, retries=1) as client:
        session = build_session(agent=_agent(), llm=client, max_budget_tokens=1200)
        await session.add_user_message("Give a short answer")
        await session.run_loop()
    assert len(requests) == 2
    assert session.phase is SessionPhase.STOPPED
    assert session.used_tokens == 1800
    assert session.state.context_tokens == 600
    assert "budget exceeded after model call" in session.state.terminal_reason
    records = _records(usage_log)
    assert len(records) == 1
    assert records[0]["usage"]["total_tokens"] == 1800


@pytest.mark.parametrize("recover", [False, True])
async def test_unknown_failure_usage_is_retained_as_unknown(recover, local_responses_server, usage_log):
    specs = [{"kind": "empty" if recover else "filter", "usage": None}]
    if recover:
        specs.append({})
    base_url, requests = local_responses_server(specs)
    async with _client("gpt-4o", base_url, retries=int(recover)) as client:
        session = build_session(agent=_agent(), llm=client)
        await session.add_user_message("Give a short answer")
        if recover:
            assert await session.run_loop() == "ok"
        else:
            with pytest.raises(ResponsesProtocolError):
                await session.run_loop()
    assert len(requests) == (2 if recover else 1)
    assert session.used_tokens == (37 if recover else 0)
    record = _records(usage_log)[0]
    assert record["usage"]["estimated"] is True
    assert record["usage"]["cached_input_tokens"] is None
    assert record["usage"]["cost_usd"] is None
    if recover:
        assert record["usage"]["raw_usage"]["attempts"] == [None, KNOWN_USAGE]


@pytest.mark.parametrize("reported", [{"input_tokens": 30}, {"input_tokens": 30, "output_tokens": 0}])
async def test_partial_and_zero_failure_counts_use_only_reported_tokens(reported, local_responses_server, usage_log):
    base_url, _requests = local_responses_server([{"kind": "filter", "usage": reported}])
    async with _client("gpt-4o", base_url) as client:
        session = build_session(agent=_agent(), llm=client)
        await session.add_user_message("Give a short answer")
        with pytest.raises(ResponsesProtocolError):
            await session.run_loop()
    assert session.used_tokens == 30
    record = _records(usage_log)[0]
    assert record["usage"]["raw_usage"] == reported
    assert record["usage"]["output_tokens"] == 0
    assert record["usage"]["estimated"] is ("output_tokens" not in reported)


@pytest.mark.parametrize("failed_summary", [False, True])
async def test_summary_attempts_and_answer_are_each_charged_once(failed_summary, local_responses_server, usage_log):
    specs = [{"kind": "filter"}] if failed_summary else [
        {"kind": "empty"}, {"text": "<summary>Earlier context.</summary>"},
    ]
    specs.append({"text": "answer"})
    base_url, requests = local_responses_server(specs)
    async with _client("gpt-4o", base_url, retries=1) as client:
        session = build_session(agent=_agent(), llm=client)
        session.messages = history()
        assert await session.run_loop() == "answer"
    assert len(requests) == (2 if failed_summary else 3)
    assert "Your task is to create a detailed summary" in requests[0]["input"][-1]["content"]
    assert session.used_tokens == (74 if failed_summary else 111)
    assert session.state.context_tokens == 30
    records = _records(usage_log)
    assert len(records) == 2
    assert [record["usage"]["total_tokens"] for record in records] == ([37, 37] if failed_summary else [74, 37])


async def _drain(session):
    while session.pending_cleanup_tasks:
        await asyncio.gather(*session.pending_cleanup_tasks, return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("trigger", ["cancel", "timeout"])
@pytest.mark.parametrize("kind", ["filter", "answer"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_late_sdk_terminal_usage_is_saved_once(
    monkeypatch, tmp_path, usage_log, summary, trigger, kind, repeat_cancel,
):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        content_type, payload = _payload(json.loads(request.content), {"kind": kind})
        return httpx.Response(200, headers={"content-type": content_type}, content=payload)

    install_sdk_transport(monkeypatch, "openai", handler)
    snapshot_path = tmp_path / "late.json"
    async with _client("gpt-4o", "http://controlled.invalid/v1") as client:
        session = build_session(
            agent=_agent(), llm=client, llm_timeout=0.01 if trigger == "timeout" else 600,
            auto_save_path=str(snapshot_path),
        )
        if summary:
            session.messages = history()
        else:
            await session.add_user_message("Give a short answer")
        session.state.wind_down_done = True
        session.runner._commit_reserve = session.max_budget_tokens - 30
        turn = asyncio.create_task(session.run_loop())
        try:
            await asyncio.wait_for(started.wait(), 1)
            if trigger == "cancel":
                turn.cancel()
                expected_error = asyncio.CancelledError
            else:
                expected_error = GenerationTimeoutError
            with pytest.raises(expected_error):
                await asyncio.wait_for(turn, 1)
            await asyncio.wait_for(cancelled.wait(), 1)
            assert session.used_tokens == 0
            if repeat_cancel:
                cancelled.clear()
                session.runner.pending_cleanup_tasks[0].cancel()
                await asyncio.wait_for(cancelled.wait(), 1)
            original_messages = session.messages.copy()
            release.set()
            await asyncio.wait_for(_drain(session), 1)
            assert session.used_tokens == 37
            assert session.runner.late_provider_usage == (37,)
            assert session.state.budget_reserve_consumed is True
            assert session.messages == original_messages
            assert session.phase is (SessionPhase.STOPPED if trigger == "cancel" else SessionPhase.ERROR)
            snapshot = SessionStore().load_snapshot(str(snapshot_path), session.agent.system_prompt)
            assert snapshot["session_state"]["used_tokens"] == 37
            records = _records(usage_log)
            assert len(records) == 1
            assert records[0]["usage"]["total_tokens"] == 37
        finally:
            release.set()
            await asyncio.gather(turn, return_exceptions=True)
            await _drain(session)


async def test_cancellation_during_retry_retains_previous_terminal_usage(
    monkeypatch, local_responses_server, usage_log,
):
    backoff_started = asyncio.Event()

    def retry_after(_error):
        backoff_started.set()
        return 60.0

    monkeypatch.setattr(retry, "extract_retry_after_seconds", retry_after)
    base_url, requests = local_responses_server([{"kind": "empty"}])
    async with _client("gpt-4o", base_url, retries=1) as client:
        session = build_session(agent=_agent(), llm=client)
        await session.add_user_message("Give a short answer")
        turn = asyncio.create_task(session.run_loop())
        await asyncio.wait_for(backoff_started.wait(), 1)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, 1)
        await asyncio.wait_for(_drain(session), 1)
    assert len(requests) == 1
    assert session.used_tokens == 37
    assert session.runner.late_provider_usage == (37,)
    records = _records(usage_log)
    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["error"]["type"] == "CancelledError"
    assert records[0]["usage"]["total_tokens"] == 37


@pytest.mark.parametrize("kind", ["filter", "answer"])
async def test_cancellation_during_ledger_write_does_not_duplicate_usage(
    monkeypatch, local_responses_server, usage_log, kind,
):
    started = threading.Event()
    release = threading.Event()
    original_record = client_module.record_api_usage

    def delayed_record(**kwargs):
        started.set()
        assert release.wait(2)
        original_record(**kwargs)

    monkeypatch.setattr(client_module, "record_api_usage", delayed_record)
    base_url, requests = local_responses_server([{"kind": kind}])
    async with _client("gpt-4o", base_url) as client:
        session = build_session(agent=_agent(), llm=client)
        await session.add_user_message("Give a short answer")
        turn = asyncio.create_task(session.run_loop())
        try:
            assert await asyncio.to_thread(started.wait, 1)
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(turn, 1)
            release.set()
            await asyncio.wait_for(_drain(session), 1)
        finally:
            release.set()
            await asyncio.gather(turn, return_exceptions=True)
            await _drain(session)
    assert len(requests) == 1
    assert session.phase is SessionPhase.STOPPED
    assert session.used_tokens == 37
    assert session.runner.late_provider_usage == (37,)
    records = _records(usage_log)
    assert len(records) == 1
    assert records[0]["status"] == ("success" if kind == "answer" else "error")
    assert records[0]["usage"]["total_tokens"] == 37
