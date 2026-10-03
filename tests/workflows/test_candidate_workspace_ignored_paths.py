from __future__ import annotations

from pathlib import Path

import pytest

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


@pytest.mark.parametrize("stage_draft", [False, True])
@pytest.mark.parametrize("draft_action", ["keep", "edit", "delete"])
@pytest.mark.asyncio
async def test_candidate_new_ignore_rule_preserves_source_draft_changes(
    tmp_path: Path, stage_draft: bool, draft_action: str,
) -> None:
    repo = _repository(tmp_path)
    draft = repo / "draft [notes].txt"
    draft.write_text("user draft\n")
    if stage_draft:
        _git(repo, "add", "--", draft.name)
    source_index = _git(repo, "ls-files", "--stage")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("ignore-source-draft")
    try:
        candidate_repo = Path(lease.candidate_workspace)
        candidate_draft = candidate_repo / draft.name
        _git(candidate_repo, "reset", "--mixed", "HEAD")
        await lease.environment.write_file(".gitignore", "*.txt\n")
        await lease.environment.write_file("new-ignored.txt", "generated output\n")
        if draft_action == "edit":
            candidate_draft.write_text("user draft\ncandidate edit\n")
        elif draft_action == "delete":
            candidate_draft.unlink()
        candidate_index = _git(candidate_repo, "ls-files", "--stage")
        source_tree = _git(candidate_repo, "rev-parse", "refs/worktree/opencollab-source")

        patch = await lease.diff()

        assert "new-ignored.txt" not in patch
        if draft_action == "keep":
            assert draft.name not in patch
        elif draft_action == "edit":
            assert "+candidate edit" in patch
            assert "deleted file mode" not in patch
        else:
            assert "deleted file mode" in patch
        assert _git(candidate_repo, "ls-files", "--stage") == candidate_index
        assert _git(candidate_repo, "rev-parse", "refs/worktree/opencollab-source") == source_tree

        await workspace.adopt(patch, preserve_paths=[draft.name] if draft_action == "keep" else [])

        assert _git(repo, "ls-files", "--stage") == source_index
        if draft_action == "delete":
            assert not draft.exists()
        else:
            expected = "user draft\ncandidate edit\n" if draft_action == "edit" else "user draft\n"
            assert draft.read_text() == expected
    finally:
        await lease.cleanup()


@pytest.mark.parametrize("commit_file", [False, True])
@pytest.mark.asyncio
async def test_candidate_captures_new_force_added_ignored_file(tmp_path: Path, commit_file: bool) -> None:
    repo = _repository(tmp_path)
    (repo / ".gitignore").write_text("*.txt\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore generated text")
    source_index = _git(repo, "ls-files", "--stage")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("force-add-ignored-file")
    try:
        candidate_repo = Path(lease.candidate_workspace)
        await lease.environment.write_file("delivered.txt", "committed content\n")
        await lease.environment.write_file("ignored-output.txt", "generated output\n")
        _git(candidate_repo, "add", "--force", "delivered.txt")
        if commit_file:
            _git(candidate_repo, "-c", "commit.gpgsign=false", "commit", "-m", "deliver ignored file")
        await lease.environment.write_file("delivered.txt", "current content\n")
        candidate_index = _git(candidate_repo, "ls-files", "--stage")

        patch = await lease.diff()

        assert "+current content" in patch
        assert "ignored-output.txt" not in patch
        assert _git(candidate_repo, "ls-files", "--stage") == candidate_index
        await workspace.adopt(patch)
        assert (repo / "delivered.txt").read_text() == "current content\n"
        assert _git(repo, "ls-files", "--stage") == source_index
    finally:
        await lease.cleanup()
