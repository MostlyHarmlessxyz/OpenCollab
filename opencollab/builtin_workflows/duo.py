"""Duo solves a task through two isolated candidates and evidence-based selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opencollab.workflows import workflow

from ._candidate_evidence_files import CandidateEvidenceFiles
from ._dual_coder import run_dual_coder
from ._file_selection import adjudicate_candidate_files
from ._prompts import _PROMPT_REVISION, SELECTION_PROMPT


def _source_evidence_root(ctx: Any, parent: str | None) -> Path | None:
    workspace = getattr(ctx, "host_workspace", None)
    if parent is None or not isinstance(workspace, str) or not workspace:
        return None
    workspace_path = Path(workspace).resolve()
    parent_path = Path(parent).resolve()
    for root in (workspace_path, *workspace_path.parents):
        if (root / ".git").exists():
            return root if parent_path.is_relative_to(root) else None
    return None


@workflow(
    name="duo",
    description="Two focused solutions with complete evidence and read-only selection",
    phases=["candidate-a", "candidate-b", "mechanical-selection", "adjudication", "adoption"],
)
async def duo(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Run Duo using task-oriented prompts and complete candidate evidence.

    ``candidate_evidence_dir`` optionally supplies a host-side parent directory.
    Each adjudication creates its own retained evidence directory.
    ``submission_mode="working_tree"`` declares caller-owned patch capture and
    later submission. The default ``"task"`` follows task-specific delivery.
    """
    evidence_parent = args.get("candidate_evidence_dir")
    evidence_root = _source_evidence_root(ctx, evidence_parent)
    files = CandidateEvidenceFiles(evidence_parent) if evidence_root is not None else None
    source_reader = None
    if files is not None:
        excluded = (files.directory.resolve().relative_to(evidence_root).as_posix(),)

        async def source_reader() -> str | None:
            source = await ctx.diff(exclude_paths=excluded)
            if source is None:
                raise RuntimeError("candidate_workspace_tracking_failure: source diff unavailable for Duo evidence")
            return source

    async def adjudicator(context: Any, **options: Any) -> tuple[str, Any, str]:
        return await adjudicate_candidate_files(
            context, **options, evidence_parent=evidence_parent, evidence_files=files,
        )

    result = await run_dual_coder(
        ctx, args, selector_prompt=SELECTION_PROMPT, adjudicator=adjudicator,
        source_reader=source_reader,
    )
    result["prompt_revision"] = _PROMPT_REVISION
    return result


__all__ = ["duo"]
