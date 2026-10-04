"""Initialize worktree submodules from already available local sources."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from opencollab.adapters._env_base import ExecResult


async def _initialize_source_available_submodules(
    source_repository: str,
    target_repository: str,
    *,
    git_in: Callable[..., Awaitable[ExecResult]],
    source_prefix: str = "",
) -> None:
    await _SourceSubmoduleTree(git_in)._initialize_source_submodule_tree(
        source_repository,
        target_repository,
        source_prefix=source_prefix,
        active_sources=set(),
    )


class _SourceSubmoduleTree:
    def __init__(self, git_in: Callable[..., Awaitable[ExecResult]]) -> None:
        self._git_in = git_in

    async def _configured_source_submodules(
        self,
        source_repository: str,
        *,
        source_prefix: str,
    ) -> list[tuple[str, str, str]]:
        modules_file = os.path.join(source_repository, ".gitmodules")
        if not os.path.isfile(modules_file):
            return []
        configured = await self._git_in(
            source_repository,
            "config",
            "--file",
            modules_file,
            "--get-regexp",
            r"^submodule\..*\.path$",
        )
        if configured.returncode == 1:
            return []
        if (
            configured.returncode != 0
            or configured.stdout_truncated
            or configured.stderr_truncated
        ):
            raise RuntimeError("cannot inspect Git submodule configuration")

        submodules: list[tuple[str, str, str]] = []
        for line in configured.stdout.splitlines():
            try:
                key, configured_path = line.split(maxsplit=1)
            except ValueError as exc:
                raise RuntimeError("invalid Git submodule path configuration") from exc
            prefix = "submodule."
            suffix = ".path"
            if not key.startswith(prefix) or not key.endswith(suffix):
                raise RuntimeError("invalid Git submodule path configuration")
            name = key[len(prefix) : -len(suffix)]
            normalized_path = configured_path.replace("\\", "/").strip("/")
            if (
                not normalized_path
                or normalized_path == ".."
                or normalized_path.startswith("../")
            ):
                raise RuntimeError("invalid Git submodule path configuration")
            if source_prefix and not (
                normalized_path == source_prefix
                or normalized_path.startswith(f"{source_prefix}/")
            ):
                continue
            source_module = os.path.realpath(
                os.path.join(source_repository, *normalized_path.split("/"))
            )
            try:
                contained = os.path.commonpath((source_repository, source_module))
            except ValueError:
                contained = ""
            if contained != source_repository or not os.path.isdir(source_module):
                raise RuntimeError(
                    f"Git submodule is not initialized in source workspace: {configured_path}"
                )
            available = await self._git_in(
                source_module,
                "rev-parse",
                "--show-toplevel",
            )
            if (
                available.returncode != 0
                or available.stdout_truncated
                or available.stderr_truncated
                or os.path.realpath(available.stdout.strip()) != source_module
            ):
                raise RuntimeError(
                    f"Git submodule is not initialized in source workspace: {configured_path}"
                )
            submodules.append((name, normalized_path, source_module))
        return submodules

    async def _initialize_source_submodule_tree(
        self,
        source_repository: str,
        target_repository: str,
        *,
        source_prefix: str,
        active_sources: set[str],
    ) -> None:
        source_repository = os.path.realpath(source_repository)
        if source_repository in active_sources:
            raise RuntimeError("cyclic initialized Git submodule source")
        active_sources.add(source_repository)
        try:
            submodules = await self._configured_source_submodules(
                source_repository,
                source_prefix=source_prefix,
            )
            for name, normalized_path, source_module in submodules:
                updated = await self._git_in(
                    target_repository,
                    "-c",
                    "protocol.allow=never",
                    "-c",
                    "protocol.file.allow=always",
                    "-c",
                    f"submodule.{name}.url={source_module}",
                    "submodule",
                    "update",
                    "--init",
                    "--no-fetch",
                    "--",
                    normalized_path,
                )
                if (
                    updated.returncode != 0
                    or updated.stdout_truncated
                    or updated.stderr_truncated
                ):
                    detail = updated.stderr.strip() or "submodule checkout failed"
                    raise RuntimeError(
                        "cannot initialize source-available submodule "
                        f"{normalized_path}: {detail}"
                    )
                target_module = os.path.realpath(
                    os.path.join(
                        target_repository,
                        *normalized_path.split("/"),
                    )
                )
                await self._initialize_source_submodule_tree(
                    source_module,
                    target_module,
                    source_prefix="",
                    active_sources=active_sources,
                )
        finally:
            active_sources.remove(source_repository)

