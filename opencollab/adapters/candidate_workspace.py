"""Environment-backed isolated candidate worktrees."""

from __future__ import annotations

import os
import posixpath
import shlex
import tempfile
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.env import DockerWorkspaceEnvironment
from opencollab.application.async_timeout import await_owned_operation
from opencollab.patches import patch_paths

CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS = 900.0


def _complete(result: Any, operation: str, allowed: tuple[int, ...] = (0,)) -> str:
    if (
        getattr(result, "returncode", -1) not in allowed
        or getattr(result, "stdout_truncated", False)
        or getattr(result, "stderr_truncated", False)
    ):
        detail = str(getattr(result, "stderr", "") or "").strip()
        raise RuntimeError(f"{operation} failed: {detail or 'incomplete command result'}")
    return str(getattr(result, "stdout", "") or "")


def _safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("candidate preserve path must be non-empty text")
    normalized = posixpath.normpath(value)
    if posixpath.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
        raise ValueError("candidate preserve path escapes the repository")
    return normalized


@asynccontextmanager
async def _candidate_temporary(
    environment: Any,
    workspace: str,
    content: str,
    *,
    prefix: str,
    suffix: str = ".tmp",
    cleanup_best_effort: bool = False,
) -> AsyncIterator[str]:
    if isinstance(environment, LocalEnvironment):
        git_directory = _complete(
            await environment.exec_cmd(
                f"git -C {shlex.quote(workspace)} rev-parse --absolute-git-dir",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate temporary storage",
        ).strip()
        # Git metadata is outside ordinary file snapshots, including linked
        # worktrees. A private directory also owns any transient Git lock files.
        with tempfile.TemporaryDirectory(prefix="opencollab-candidate-", dir=git_directory) as directory:
            owner = LocalEnvironment(directory)
            try:
                yield await owner.write_temp_file(content, prefix=prefix, suffix=suffix)
            finally:
                try:
                    await await_owned_operation(owner.cleanup(), propagate_cancellation=True)
                except Exception:
                    if not cleanup_best_effort:
                        raise
    else:
        path = await environment.write_temp_file(content, prefix=prefix, suffix=suffix)
        try:
            yield path
        finally:
            try:
                await await_owned_operation(environment.remove_file(path), propagate_cancellation=True)
            except Exception:
                if not cleanup_best_effort:
                    raise


async def _raw_diff(environment: Any, exclude_paths: Sequence[str] = ()) -> str:
    workspace = getattr(environment, "workspace", None)
    if not isinstance(workspace, str) or not workspace:
        raise RuntimeError("candidate diff workspace is unavailable")
    return await _raw_diff_at(environment, workspace, exclude_paths)


async def _raw_diff_at(
    environment: Any,
    workspace: str,
    exclude_paths: Sequence[str] = (),
    *,
    base_revision: str = "HEAD",
    index_file: str | None = None,
) -> str:
    excluded = tuple(_safe_path(path) for path in exclude_paths)
    pathspec = ""
    if excluded:
        pathspec = " -- . " + " ".join(
            shlex.quote(f":(exclude){path}") for path in excluded
        )
    git = f"git -C {shlex.quote(workspace)}"
    cached = ""
    if index_file is not None:
        git = f"GIT_INDEX_FILE={shlex.quote(index_file)} {git}"
        cached = " --cached"
    tracked = _complete(
        await environment.exec_cmd(
            f"{git} --no-pager diff{cached} {shlex.quote(base_revision)} --binary --no-ext-diff"
            + pathspec,
            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
        ),
        "candidate tracked diff",
    )
    untracked = _complete(
        await environment.exec_cmd(
            f"{git} ls-files --others --exclude-standard -z"
            + pathspec,
            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
        ),
        "candidate untracked listing",
    )
    parts = [tracked.rstrip("\n")] if tracked.strip() else []
    for path in (item for item in untracked.split("\0") if item):
        patch = _complete(
            await environment.exec_cmd(
                "git -C "
                f"{shlex.quote(workspace)} --no-pager diff --no-index --binary "
                f"--no-ext-diff -- /dev/null {shlex.quote(path)}",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            f"candidate untracked diff for {path}",
            allowed=(0, 1),
        )
        if patch.strip():
            parts.append(patch.rstrip("\n"))
    return "\n".join(parts) + ("\n" if parts else "")


@dataclass(slots=True)
class _CandidateLease:
    base_environment: Any
    environment: Any
    source_workspace: str
    candidate_workspace: str
    base_revision: str
    cleaned: bool = False

    async def diff(self, exclude_paths: Sequence[str] = ()) -> str:
        async with _candidate_temporary(
            self.base_environment, self.source_workspace,
            "", prefix=".candidate-capture-index-",
        ) as index_file:
            git = f"GIT_INDEX_FILE={shlex.quote(index_file)} git -C {shlex.quote(self.candidate_workspace)}"
            # Stage the current contents in an owned temporary index. Candidate
            # commits and index resets leave the same delivered file changes.
            _complete(
                await self.base_environment.exec_cmd(
                    f"{git} read-tree HEAD && {git} add --all -- .",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate contents capture",
            )
            original_git = f"git -C {shlex.quote(self.candidate_workspace)}"
            known = _complete(
                await self.base_environment.exec_cmd(
                    f"{original_git} ls-tree -r --name-only -z {shlex.quote(self.base_revision)} "
                    f"&& {original_git} ls-files --cached -z",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate known paths",
            )
            remaining = _complete(
                await self.base_environment.exec_cmd(
                    f"{git} ls-files --others -z",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate remaining files",
            )
            # Retain existing source and staged files when new ignore rules hide
            # them. The remaining listing supplies current files, so deletions
            # and file/directory replacements follow their actual contents.
            omitted = sorted(set(known.split("\0")).intersection(remaining.split("\0")) - {""})
            if omitted:
                async with _candidate_temporary(
                    self.base_environment, self.source_workspace,
                    "".join(f":(literal){path}\0" for path in omitted),
                    prefix=".candidate-capture-paths-",
                ) as paths_file:
                    _complete(
                        await self.base_environment.exec_cmd(
                            f"{git} add --force --pathspec-from-file={shlex.quote(paths_file)} --pathspec-file-nul",
                            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                        ),
                        "candidate known files capture",
                    )
            return await _raw_diff_at(
                self.base_environment,
                self.candidate_workspace,
                exclude_paths,
                base_revision=self.base_revision,
                index_file=index_file,
            )

    async def cleanup(self) -> None:
        if self.cleaned:
            return
        await self.environment.cleanup()
        result = await self.base_environment.exec_cmd(
            "git -C "
            f"{shlex.quote(self.source_workspace)} worktree remove --force -- "
            f"{shlex.quote(self.candidate_workspace)}",
            timeout=120,
        )
        _complete(result, "candidate worktree cleanup")
        self.cleaned = True


class EnvCandidateWorkspace:
    """Candidate workspace port implemented over one task environment."""

    def __init__(self, environment: Any, *, workspace: str | None = None) -> None:
        self._environment = environment
        self._workspace = workspace or getattr(environment, "workspace", None)
        if not isinstance(self._workspace, str) or not self._workspace:
            raise ValueError("candidate source workspace is unavailable")

    async def _candidate_environment(self, path: str) -> Any:
        if isinstance(self._environment, LocalEnvironment):
            return LocalEnvironment(path)
        if isinstance(self._environment, DockerEnvironment):
            await self._environment.setup()
            container_id = self._environment._container_id
            if not container_id:
                raise RuntimeError("candidate container identity is unavailable")
            return DockerWorkspaceEnvironment(
                container_id=container_id,
                repo_root=path,
                command_prefix=self._environment._command_prefix,
                timeout_returncode=self._environment._timeout_returncode,
            )
        raise RuntimeError("candidate worktrees require a local or Docker environment")

    async def acquire(self, label: str) -> _CandidateLease:
        token = uuid.uuid4().hex
        if isinstance(self._environment, LocalEnvironment):
            path = tempfile.mkdtemp(prefix="opencollab-candidate-")
            os.rmdir(path)
        else:
            path = f"/tmp/opencollab-candidate-{token}"
        base_revision = _complete(
            await self._environment.exec_cmd(
                f"git -C {shlex.quote(self._workspace)} rev-parse --verify HEAD",
                timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
            ),
            "candidate base revision",
        ).strip()
        source_patch = await _raw_diff_at(
            self._environment, self._workspace, base_revision=base_revision,
        )
        result = await self._environment.exec_cmd(
            "git -C "
            f"{shlex.quote(self._workspace)} worktree add --detach -- "
            f"{shlex.quote(path)} {shlex.quote(base_revision)}",
            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
        )
        _complete(result, f"candidate worktree setup for {label}")
        environment = await self._candidate_environment(path)
        await environment.setup()
        lease = _CandidateLease(
            base_environment=self._environment,
            environment=environment,
            source_workspace=self._workspace,
            candidate_workspace=path,
            base_revision=base_revision,
        )
        try:
            if source_patch.strip():
                async with _candidate_temporary(
                    self._environment, self._workspace, source_patch,
                    prefix=".candidate-source-", suffix=".patch",
                ) as source_file:
                    _complete(
                        await self._environment.exec_cmd(
                            f"git -C {shlex.quote(path)} apply --index --binary --whitespace=nowarn "
                            f"-- {shlex.quote(source_file)}",
                            timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                        ),
                        "candidate source contents",
                    )
                # The candidate's index records its starting contents, including
                # source untracked files, while the source index stays untouched.
                lease.base_revision = _complete(
                    await self._environment.exec_cmd(
                        f"git -C {shlex.quote(path)} write-tree",
                        timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                    ),
                    "candidate source tree",
                ).strip()
            # A worktree-local ref retains the original contents for recovery
            # throughout the lease, including after commits or Git collection.
            _complete(
                await self._environment.exec_cmd(
                    f"git -C {shlex.quote(path)} update-ref refs/worktree/opencollab-source "
                    f"{shlex.quote(lease.base_revision)}",
                    timeout=CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
                ),
                "candidate source recovery reference",
            )
            return lease
        except BaseException:
            await lease.cleanup()
            raise

    async def source_diff(self, exclude_paths: Sequence[str] = ()) -> str:
        return await _raw_diff_at(
            self._environment,
            self._workspace,
            exclude_paths,
        )

    async def restore_source(self, patch: str) -> None:
        await self._replace_source_diff(patch)

    async def _replace_source_diff(self, patch: str) -> None:
        current = await _raw_diff_at(self._environment, self._workspace)
        async with _candidate_temporary(
            self._environment, self._workspace, current,
            prefix=".candidate-current-", suffix=".patch", cleanup_best_effort=True,
        ) as current_file:
            async with _candidate_temporary(
                self._environment, self._workspace, patch,
                prefix=".candidate-target-", suffix=".patch", cleanup_best_effort=True,
            ) as target_file:
                reversed_current = False
                try:
                    if current.strip():
                        await self._apply(
                            current_file,
                            "candidate current-source removal",
                            reverse=True,
                        )
                        reversed_current = True
                    if patch.strip():
                        await self._apply(target_file, "candidate source restoration")
                except BaseException as failure:
                    if reversed_current:
                        try:
                            await self._apply(
                                current_file,
                                "candidate source restoration rollback",
                            )
                        except BaseException as rollback:
                            raise RuntimeError(
                                "candidate source restoration and rollback both failed"
                            ) from rollback
                    raise failure

    async def _apply(self, path: str, operation: str, *, reverse: bool = False) -> None:
        reverse_flag = " --reverse" if reverse else ""
        result = await self._environment.exec_cmd(
            "git -C "
            f"{shlex.quote(self._workspace)} apply --binary --whitespace=nowarn"
            f"{reverse_flag} -- {shlex.quote(path)}",
            timeout=120,
        )
        _complete(result, operation)

    async def _patch_paths(self, path: str) -> set[str]:
        result = await self._environment.exec_cmd(
            "git -C "
            f"{shlex.quote(self._workspace)} apply --numstat -z -- {shlex.quote(path)}",
            timeout=30,
        )
        output = _complete(result, "candidate path inspection")
        return {
            line.split("\t", 2)[-1]
            for line in output.split("\0")
            if line.count("\t") >= 2
        }

    async def adopt(
        self,
        patch: str,
        preserve_paths: Sequence[str] = (),
    ) -> None:
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError("candidate patch must be non-empty")
        preserved = tuple(_safe_path(path) for path in preserve_paths)
        async with _candidate_temporary(
            self._environment, self._workspace, patch,
            prefix=".candidate-adopt-", suffix=".patch", cleanup_best_effort=True,
        ) as candidate_file:
            candidate_paths = await self._patch_paths(candidate_file)
            candidate_paths.update(patch_paths(patch))
            if candidate_paths.intersection(preserved):
                raise ValueError("candidate patch overlaps a preserved path")
            # Git applies the candidate increment atomically to the current
            # files. Independent user edits survive and conflicts leave the
            # source contents and index in place.
            await self._apply(candidate_file, "candidate adoption")


__all__ = ["EnvCandidateWorkspace"]
