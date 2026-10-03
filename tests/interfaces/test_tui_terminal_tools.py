"""Terminal lifecycle events settle the matching agent's active indicators."""

import pytest

from opencollab.adapters.tui import TUI
from opencollab.domain.events import SchedulerEvent, SessionRuntimeEvent


@pytest.mark.parametrize("terminal", ["agent_cancelled", "agent_failed", "agent_completed"])
def test_terminal_agent_event_clears_child_tools_and_thinking(terminal):
    tui = TUI()
    tui.set_redraw(lambda: None)
    tui.event_handler(SchedulerEvent("agent_spawned", {"aid": 1, "role": "coder", "parent_aid": 0}))
    tui.event_handler(SessionRuntimeEvent("tool_start", {"aid": 0, "tool": "spawn_agent"}))
    for tool_id in ("first", "second"):
        tui.event_handler(SessionRuntimeEvent("tool_start", {
            "aid": 1, "tool": "bash", "tool_call_id": tool_id,
            "args": {"command": "fixture command"},
        }))
    tui.event_handler(SessionRuntimeEvent("step_start", {"aid": 1, "step": 2}))
    child = tui._state_for(1)
    assert len(child.active_tools) == 3
    assert child.thinking is not None

    tui.event_handler(SchedulerEvent(terminal, {"aid": 1, "role": "coder"}))
    tui.select_agent(1)

    assert child.active_tools == {}
    assert child.thinking is None
    assert tui._state_for(0).active_tools
    assert "fixture command" not in (tui.hud_ansi() or "")
    tui.event_handler(SchedulerEvent("agent_resumed", {"aid": 1, "role": "coder"}))
    tui.event_handler(SessionRuntimeEvent("tool_start", {
        "aid": 1, "tool": "file_read", "tool_call_id": "fresh", "args": {"path": "a.py"},
    }))
    assert list(child.active_tools) == ["A1:file_read#fresh"]
