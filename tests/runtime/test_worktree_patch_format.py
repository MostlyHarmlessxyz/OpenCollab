"""Exported worktree patches remain applicable under Git display settings."""

import subprocess
from pathlib import Path

import pytest

from opencollab.adapters.env import WorktreeEnvironment


@pytest.mark.parametrize("git_repo", [False, True])
@pytest.mark.parametrize("settings", [
    {}, {"diff.noprefix": "true"}, {"color.ui": "always"}, {"diff.context": "0"},
    {"diff.srcPrefix": "before/custom/", "diff.dstPrefix": "after/custom/", "color.diff": "always"},
])
async def test_exported_patch_applies_with_display_settings(tmp_path, monkeypatch, git_repo, settings):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.py").write_text("alpha\nbeta\ngamma\ndelta\n")
    (source / "binary.bin").write_bytes(b"\x00\xfforiginal\n")

    def git(*args, **kwargs):
        return subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True, check=True, **kwargs)

    global_config = tmp_path / "gitconfig"
    global_config.touch()
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    if git_repo:
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    for key, value in settings.items():
        git("config", "--local" if git_repo else "--global", key, value)

    env = WorktreeEnvironment(str(source))
    try:
        await env.setup()
        await env.write_file("source.py", "alpha\nBETA\ngamma\ndelta\n")
        (Path(env.workspace) / "binary.bin").write_bytes(b"\x00\xffupdated\n")
        await env.write_file("new.txt", "new contents\n")
        patch = await env.get_diff()
        assert "\x1b[" not in patch
        assert env.workspace not in patch
        assert "GIT binary patch" in patch
        git("apply", "--check", "-", input=patch)
        git("apply", "-", input=patch)
        assert (source / "source.py").read_text() == "alpha\nBETA\ngamma\ndelta\n"
        assert (source / "binary.bin").read_bytes() == b"\x00\xffupdated\n"
        assert (source / "new.txt").read_text() == "new contents\n"
        for key, value in settings.items():
            assert git("config", "--get", key).stdout.strip() == value
    finally:
        await env.cleanup()


@pytest.mark.parametrize("binary", [False, True])
async def test_directory_copy_rename_exports_an_applicable_patch(tmp_path, monkeypatch, binary):
    source = tmp_path / "source"
    source.mkdir()
    before_name, after_name = "before name.txt", "after name.txt"
    payload = b"\x00\xffrename payload\n" if binary else b"ordinary rename contents\n"
    (source / before_name).write_bytes(payload)
    config = tmp_path / "gitconfig"
    config.write_text("[diff]\n\trenames = true\n")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    env = WorktreeEnvironment(str(source))
    try:
        await env.setup()
        workspace = Path(env.workspace)
        (workspace / before_name).rename(workspace / after_name)
        patch = await env.get_diff()
        assert env.workspace not in patch
        assert "opencollab-cp-baseline-" not in patch
        for arguments in [("apply", "--check", "-"), ("apply", "-")]:
            subprocess.run(
                ["git", "-C", str(source), *arguments], input=patch,
                text=True, capture_output=True, check=True,
            )
        assert not (source / before_name).exists()
        assert (source / after_name).read_bytes() == payload
    finally:
        await env.cleanup()
