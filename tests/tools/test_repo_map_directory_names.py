"""Generated directory names may also name useful regular entry files."""

import asyncio

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.repo_map import build_repo_map, build_repo_map_via_env


@pytest.mark.parametrize("via_environment", [False, True])
@pytest.mark.xfail(strict=True, reason="OC-D11 directory names also exclude regular files")
def test_generated_directory_rule_keeps_same_named_regular_files(tmp_path, via_environment):
    names = ("build", "dist", "venv", "node_modules", "__pycache__")
    for name in names:
        (tmp_path / name).write_text("regular entry file\n", encoding="utf-8")
        generated = tmp_path / "src" / name
        generated.mkdir(parents=True)
        (generated / "generated.js").write_text("generated\n", encoding="utf-8")
    (tmp_path / "build.sh").write_text("entry\n", encoding="utf-8")
    (tmp_path / ".env").write_text("hidden\n", encoding="utf-8")
    if via_environment:
        result = asyncio.run(build_repo_map_via_env(LocalEnvironment(str(tmp_path))))
    else:
        result = build_repo_map(str(tmp_path))

    entries = {line.strip() for line in result.splitlines()}
    for name in names:
        assert name in entries
    assert "build.sh" in entries
    assert "generated.js" not in result
    assert ".env" not in result
