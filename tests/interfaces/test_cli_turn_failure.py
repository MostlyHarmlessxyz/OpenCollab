"""CLI terminal errors preserve output, cleanup, and interactive controls."""

import contextlib
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

import opencollab.adapters.cli.main as cli_main
from opencollab.application.scheduler_types import SchedulerTurnError
from opencollab.domain.session import SessionPhase
from tests.interfaces.test_cli_main_lifecycle import FakeTracer, config, install_cli_fakes


@pytest.mark.parametrize("prompt_source", ["prompt", "prompt-file"])
def test_cli_error_turn_has_nonzero_exit_after_output_and_cleanup(
    monkeypatch, tmp_path, prompt_source,
):
    events = []

    class Scheduler:
        used_tokens = 1
        lead_session = SimpleNamespace(auto_save_path=None, step_count=1)

        def team_roster(self):
            return []

        async def run_turn(self, aid, line, *, cancel_event=None):
            raise SchedulerTurnError(aid, SessionPhase.ERROR, "fixture failure", "partial answer")

        def agent_step_count(self, aid):
            return 1

        async def cleanup(self):
            events.append("cleanup")

    tracer = FakeTracer()
    install_cli_fakes(monkeypatch, Scheduler(), tracer)
    monkeypatch.setattr(cli_main, "resolve_config", lambda *args: {**config(), "api_key": None, "base_url": None})
    monkeypatch.setattr(cli_main, "missing_api_key_for", lambda *args: False)
    monkeypatch.setattr(cli_main, "_report_turn_failure", lambda *args: events.append("report"))
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: False, raising=False)
    if prompt_source == "prompt-file":
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("do work", encoding="utf-8")
        arguments = ["--prompt-file", str(prompt_file)]
    else:
        arguments = ["--prompt", "do work"]

    result = CliRunner().invoke(cli_main.app, [*arguments, "--workspace", str(tmp_path), "--yolo"])

    assert events == ["report", "cleanup"]
    assert tracer.closed is True
    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_cli_interactive_error_turn_accepts_a_followup(monkeypatch, tmp_path):
    calls = []

    class Scheduler:
        used_tokens = 1
        lead_session = SimpleNamespace(auto_save_path=None, step_count=1)

        def team_roster(self):
            return []

        async def run_turn(self, aid, line, *, cancel_event=None):
            calls.append(line)
            if len(calls) == 1:
                raise SchedulerTurnError(aid, SessionPhase.ERROR, "fixture failure", None)

        def agent_step_count(self, aid):
            return 1

        async def cleanup(self):
            calls.append("cleanup")

    async def read_followup(prompt, queue, tui, lead, **kwargs):
        queue.submit("first", 0)
        await queue.drain()
        queue.submit("followup", 0)
        await queue.drain()

    tracer = FakeTracer()
    install_cli_fakes(monkeypatch, Scheduler(), tracer)
    monkeypatch.setattr(cli_main, "_read_loop", read_followup)
    await cli_main._run(str(tmp_path), config(), None, True, True, False)

    assert calls == ["first", "followup", "cleanup"]
    assert tracer.closed is True


@pytest.mark.asyncio
async def test_cli_hold_preserves_inspection_before_error_exit(monkeypatch, tmp_path):
    events = []

    class Scheduler:
        used_tokens = 1
        lead_session = SimpleNamespace(auto_save_path=None, step_count=1)

        def team_roster(self):
            return []

        async def run_turn(self, aid, line, *, cancel_event=None):
            raise SchedulerTurnError(aid, SessionPhase.ERROR, "fixture failure", None)

        def agent_step_count(self, aid):
            return 1

        async def cleanup(self):
            events.append("cleanup")

    class Prompt:
        def __init__(self, *args, **kwargs):
            pass

        def attached(self):
            return contextlib.nullcontext()

        async def ask(self, *args, **kwargs):
            raise AssertionError("no prompt input needed")

    async def inspect_then_exit(prompt, queue, tui, lead, **kwargs):
        await queue.drain()
        events.append("inspect")

    tracer = FakeTracer()
    install_cli_fakes(monkeypatch, Scheduler(), tracer)
    monkeypatch.setattr(cli_main, "LivePrompt", Prompt)
    monkeypatch.setattr(cli_main, "_read_loop", inspect_then_exit)
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_main.sys.stdout, "isatty", lambda: True, raising=False)

    with pytest.raises(typer.Exit) as error:
        await cli_main._run(
            str(tmp_path), config(), None, True, True, False,
            one_shot_prompt="do work", hold_after_run=True,
        )

    assert error.value.exit_code == 1
    assert events == ["inspect", "cleanup"]
    assert tracer.closed is True
