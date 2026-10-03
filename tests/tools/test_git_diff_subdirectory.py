"""Subdirectory workspaces preserve tracked and untracked patch content."""

import asyncio
import subprocess

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.git_diff import GitDiffTool
from opencollab.application.tool_execution import ToolRuntime


@pytest.mark.parametrize("stat_only", [False, True])
@pytest.mark.parametrize("focused", [False, True])
@pytest.mark.xfail(strict=True, reason="P3-05 untracked paths are relative to the Git root")
def test_git_diff_reads_untracked_paths_from_a_unicode_subdirectory(tmp_path, stat_only, focused):
    root = tmp_path / "root repo"
    workspace = root / "子 package"
    workspace.mkdir(parents=True)

    def git(*arguments):
        subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "t@example.com")
    tracked = workspace / "tracked.py"
    tracked.write_text("original = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    tracked.write_text("tracked_change = 2\n", encoding="utf-8")
    (workspace / "新 file.py").write_text("new_change = 3\n", encoding="utf-8")
    (root / "outside.py").write_text("outside_change = 4\n", encoding="utf-8")
    parameters = {"include_status": False, "stat_only": stat_only}
    if focused:
        parameters["path"] = "新 file.py"
    runtime = ToolRuntime(environment=LocalEnvironment(str(workspace)), safety_policy=None, permission_policy=None)

    result = asyncio.run(GitDiffTool().execute_with_runtime(parameters, runtime))

    assert "untracked diff unavailable" not in result
    assert "file.py" in result
    if stat_only:
        assert "1 insertion" in result
        if not focused:
            assert "tracked.py" in result
    else:
        assert "+new_change = 3" in result
        assert ("+tracked_change = 2" in result) is not focused
        assert ("+outside_change = 4" in result) is not focused
