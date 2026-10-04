from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.adapters.env import LocalEnvironment
from opencollab.application.workflow import WorkflowContext
from tests.support.workflow_context_test_support import (
    CancelCleanupSession,
    FakeFactory,
    FakeSession,
)


def _repository(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    (root / "source.txt").write_text("source\n")
    subprocess.run(["git", "-C", str(root), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
    return root


def _context(repository: Path, sessions: list[FakeSession], concurrency: int) -> WorkflowContext:
    environment = LocalEnvironment(str(repository))
    return WorkflowContext(
        FakeFactory(sessions),
        max_concurrency=concurrency,
        task_concurrency=concurrency,
        candidate_workspace=EnvCandidateWorkspace(environment),
    )


@pytest.mark.asyncio
async def test_parallel_candidate_children_share_parent_agent_session_cap(tmp_path: Path) -> None:
    running = 0
    high_water = 0
    admitted = asyncio.Event()
    candidates_started = 0
    both_candidates_started = asyncio.Event()
    release = asyncio.Event()

    async def entered() -> None:
        nonlocal running, high_water
        running += 1
        high_water = max(high_water, running)
        if running >= 2:
            admitted.set()
        try:
            await release.wait()
        finally:
            running -= 1

    sessions = [FakeSession(on_enter=entered) for _ in range(4)]
    parent = _context(_repository(tmp_path / "repo"), sessions, concurrency=2)

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        nonlocal candidates_started
        candidates_started += 1
        if candidates_started == 2:
            both_candidates_started.set()
        await child.parallel([lambda: child.agent("one"), lambda: child.agent("two")])
        return "done"

    task = asyncio.create_task(parent.parallel([
        lambda: parent.candidate_workflow(nested, {}, label="A"),
        lambda: parent.candidate_workflow(nested, {}, label="B"),
    ]))
    try:
        await asyncio.wait_for(both_candidates_started.wait(), timeout=5)
        await asyncio.wait_for(admitted.wait(), timeout=5)
        for _ in range(20):
            await asyncio.sleep(0)
        assert high_water == 2
    finally:
        release.set()
    results = await asyncio.wait_for(task, timeout=10)

    assert [result.output for result in results] == ["done", "done"]
    assert high_water == 2


@pytest.mark.asyncio
async def test_candidate_child_parallelism_uses_available_cap_without_deadlock(tmp_path: Path) -> None:
    running = 0
    high_water = 0
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def entered() -> None:
        nonlocal running, high_water
        running += 1
        high_water = max(high_water, running)
        if running == 2:
            admitted.set()
        try:
            await release.wait()
        finally:
            running -= 1

    parent = _context(
        _repository(tmp_path / "repo"),
        [FakeSession(on_enter=entered) for _ in range(2)],
        concurrency=2,
    )

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        await child.parallel([lambda: child.agent("one"), lambda: child.agent("two")])
        return "done"

    task = asyncio.create_task(parent.candidate_workflow(nested, {}, label="candidate"))
    try:
        await asyncio.wait_for(admitted.wait(), timeout=5)
        assert high_water == 2
    finally:
        release.set()

    result = await asyncio.wait_for(task, timeout=10)
    assert result.output == "done"
    assert high_water == 2


@pytest.mark.asyncio
async def test_candidate_child_finishes_with_single_session_slot(tmp_path: Path) -> None:
    parent = _context(
        _repository(tmp_path / "repo"),
        [FakeSession(), FakeSession()],
        concurrency=1,
    )

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        results = await child.parallel([lambda: child.agent("one"), lambda: child.agent("two")])
        return ",".join(str(result) for result in results)

    result = await asyncio.wait_for(
        parent.candidate_workflow(nested, {}, label="candidate"),
        timeout=5,
    )

    assert result.output == "done,done"


@pytest.mark.asyncio
async def test_parent_agent_overlaps_candidate_child_within_shared_cap(tmp_path: Path) -> None:
    running = 0
    high_water = 0
    both_entered = asyncio.Event()
    parent_entered = asyncio.Event()
    release = asyncio.Event()

    async def entered() -> None:
        nonlocal running, high_water
        running += 1
        high_water = max(high_water, running)
        if running == 2:
            both_entered.set()
        try:
            await release.wait()
        finally:
            running -= 1

    async def parent_session_entered() -> None:
        parent_entered.set()
        await entered()

    parent = _context(
        _repository(tmp_path / "repo"),
        [FakeSession(on_enter=parent_session_entered), FakeSession(on_enter=entered)],
        concurrency=2,
    )

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        return str(await child.agent("candidate child"))

    parent_task = asyncio.create_task(parent.agent("parent agent"))
    candidate_task = None
    try:
        await asyncio.wait_for(parent_entered.wait(), timeout=5)
        candidate_task = asyncio.create_task(
            parent.candidate_workflow(nested, {}, label="candidate")
        )
        await asyncio.wait_for(both_entered.wait(), timeout=5)
        for _ in range(10):
            await asyncio.sleep(0)
        assert high_water == 2
    finally:
        release.set()
        pending = [task for task in (parent_task, candidate_task) if task is not None]
        await asyncio.gather(*pending, return_exceptions=True)

    assert await parent_task == "done"
    assert candidate_task is not None
    assert (await candidate_task).output == "done"
    assert high_water == 2


@pytest.mark.asyncio
async def test_timed_out_candidate_child_keeps_single_slot_until_cleanup_finishes(
    tmp_path: Path,
) -> None:
    child_session = CancelCleanupSession()
    parent_started = asyncio.Event()

    async def parent_entered() -> None:
        parent_started.set()

    parent_session = FakeSession(on_enter=parent_entered)
    factory = FakeFactory([child_session, parent_session])
    repository = _repository(tmp_path / "repo")
    environment = LocalEnvironment(str(repository))
    parent = WorkflowContext(
        factory,
        max_concurrency=1,
        task_concurrency=1,
        candidate_workspace=EnvCandidateWorkspace(environment),
    )
    candidate_returned_from_agent = asyncio.Event()

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        await child.agent("slow child", timeout=0.01)
        candidate_returned_from_agent.set()
        return "done"

    candidate_task = asyncio.create_task(
        parent.candidate_workflow(nested, {}, label="candidate")
    )
    parent_task = None
    try:
        await asyncio.wait_for(candidate_returned_from_agent.wait(), timeout=5)
        await asyncio.wait_for(child_session.cancel_seen.wait(), timeout=5)
        parent_task = asyncio.create_task(parent.agent("parent waits for child cleanup"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(factory.builds) == 1
        assert not parent_started.is_set()
    finally:
        child_session.release_cancel.set()
        pending = [task for task in (candidate_task, parent_task) if task is not None]
        await asyncio.gather(*pending, return_exceptions=True)
        await parent.wait_for_pending_cleanup()

    assert candidate_task.result().output == "done"
    assert parent_task is not None
    assert await parent_task == "done"
    assert parent_started.is_set()
    assert len(factory.builds) == 2
