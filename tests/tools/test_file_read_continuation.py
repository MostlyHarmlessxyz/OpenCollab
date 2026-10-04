"""Character cuts and line pagination need distinct continuation advice."""

import asyncio

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.application.tool_execution import ToolRuntime


def test_long_single_line_read_explains_character_cut_without_next_line(tmp_path):
    (tmp_path / "minified.js").write_text("x" * 200, encoding="utf-8")
    runtime = ToolRuntime(environment=LocalEnvironment(str(tmp_path)), safety_policy=None, permission_policy=None)

    result = asyncio.run(FileReadTool(max_read_chars=100).execute_with_runtime({"path": "minified.js"}, runtime))

    assert "character" in result
    assert "max_read_chars" in result
    assert "more lines below" not in result
    assert "offset=2" not in result


def test_complete_line_window_keeps_working_line_pagination(tmp_path):
    (tmp_path / "lines.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")
    runtime = ToolRuntime(environment=LocalEnvironment(str(tmp_path)), safety_policy=None, permission_policy=None)
    tool = FileReadTool(max_read_chars=100)

    first = asyncio.run(tool.execute_with_runtime({"path": "lines.txt", "limit": 1}, runtime))
    second = asyncio.run(tool.execute_with_runtime({"path": "lines.txt", "offset": 2, "limit": 1}, runtime))

    assert "offset=2" in first
    assert "2\tsecond" in second
    assert "offset=3" in second
