"""Retain candidates whose submodule changes cannot be delivered as a patch."""

from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.application.workflow import WorkflowContext
from opencollab.application.workflow_candidates import CandidateCaptureError
from tests.workflows.test_candidate_workspace_submodules import _dependency
from tests.workflows.test_workflow_candidate_workspace import _EditingSession, _Factory, _git, _repository


def _source_with_dependency(tmp_path, nested):
    leaf = _dependency(tmp_path, "leaf")
    dependency = leaf
    module_path = "vendor/dependency"
    if nested:
        middle = _dependency(tmp_path, "middle")
        _git(middle, "-c", "protocol.file.allow=always", "submodule", "add", str(leaf), "vendor/leaf")
        _git(middle, "commit", "-am", "record nested module")
        dependency = middle
        module_path += "/vendor/leaf"
    source = _repository(tmp_path)
    _git(source, "-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor/dependency")
    if nested:
        _git(
            source / "vendor/dependency", "-c", "protocol.file.allow=always",
            "submodule", "update", "--init", "--no-fetch", "--", "vendor/leaf",
        )
    _git(source, "commit", "-am", "record dependency")
    return source, module_path


@pytest.mark.parametrize("mode", ["agent", "workflow"])
@pytest.mark.parametrize("mutation", ["dirty", "untracked", "commit"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("edit_root", [False, True])
async def test_candidate_submodule_changes_retain_complete_worktree(tmp_path, mode, mutation, nested, edit_root):
    source, module_path = _source_with_dependency(tmp_path, nested)
    environment = LocalEnvironment(str(source))
    backend = EnvCandidateWorkspace(environment)
    leases = []
    acquire = backend.acquire

    async def capture_lease(label):
        lease = await acquire(label)
        leases.append(lease)
        return lease

    backend.acquire = capture_lease

    def edit(root):
        module = root / module_path
        # Explicit ignore settings must not conceal changes from capture.
        _git(root, "config", "submodule.vendor/dependency.ignore", "all")
        if nested:
            _git(root / "vendor/dependency", "config", "submodule.vendor/leaf.ignore", "all")
        if mutation == "untracked":
            (module / "new_source.py").write_text("value = 2\n")
        else:
            (module / "source.py").write_text("value = 2\n")
            if mutation == "commit":
                _git(module, "config", "user.name", "Test User")
                _git(module, "config", "user.email", "test@example.invalid")
                _git(module, "commit", "-am", "update dependency")
        if edit_root:
            (root / "source.py").write_text("value = 2\n")

    class Session(_EditingSession):
        async def run_loop(self, _cancel_event=None):
            edit(Path(self.environment.workspace))
            return "dependency checked"

    class Factory(_Factory):
        def build_workflow_session(self, **kwargs):
            return Session(kwargs["env"])

    context = WorkflowContext(Factory(environment), candidate_workspace=backend)

    async def nested_workflow(child, _args):
        edit(Path(child.workspace_root))
        return "dependency checked"

    try:
        with pytest.raises(CandidateCaptureError) as failure:
            if mode == "agent":
                await context.candidate_agent("repair dependency", label="dependency")
            else:
                await context.candidate_workflow(nested_workflow, {}, label="dependency")
        assert "submodule changes" in str(failure.value.__cause__)
        lease = leases[0]
        candidate = Path(lease.candidate_workspace)
        assert candidate.is_dir()
        assert str(candidate) in str(failure.value)
        assert lease.cleaned is False
        changed_file = "new_source.py" if mutation == "untracked" else "source.py"
        assert (candidate / module_path / changed_file).read_text() == "value = 2\n"
        assert (candidate / "source.py").read_text() == ("value = 2\n" if edit_root else "value = 1\n")
        assert (source / "source.py").read_text() == "value = 1\n"
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
        assert not (source / module_path / "new_source.py").exists()
        assert context.budget._leases == []
    finally:
        for lease in leases:
            await lease.cleanup()
        await environment.cleanup()


@pytest.mark.parametrize("mutation", ["dirty", "untracked", "commit"])
@pytest.mark.parametrize("nested", [False, True])
async def test_changed_source_submodule_is_preserved_before_acquisition(tmp_path, mutation, nested):
    source, module_path = _source_with_dependency(tmp_path, nested)
    module = source / module_path
    _git(source, "config", "submodule.vendor/dependency.ignore", "all")
    if nested:
        _git(source / "vendor/dependency", "config", "submodule.vendor/leaf.ignore", "all")
    name = "new_source.py" if mutation == "untracked" else "source.py"
    (module / name).write_text("value = 2\n")
    if mutation == "commit":
        _git(module, "config", "user.name", "Test User")
        _git(module, "config", "user.email", "test@example.invalid")
        _git(module, "commit", "-am", "update source dependency")
    (source / "source.py").write_text("value = 3\n")
    environment = LocalEnvironment(str(source))
    try:
        with pytest.raises(RuntimeError, match="submodule changes"):
            await EnvCandidateWorkspace(environment).acquire("changed-source")
        assert (module / name).read_text() == "value = 2\n"
        assert (source / "source.py").read_text() == "value = 3\n"
        assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1
    finally:
        await environment.cleanup()


@pytest.mark.parametrize("replacement", [False, True])
async def test_removed_or_replaced_submodule_is_retained(tmp_path, replacement):
    source, module_path = _source_with_dependency(tmp_path, False)
    environment = LocalEnvironment(str(source))
    lease = await EnvCandidateWorkspace(environment).acquire("remove-dependency")
    candidate = Path(lease.candidate_workspace)
    try:
        _git(candidate, "rm", "-f", module_path)
        if replacement:
            module = candidate / module_path
            module.mkdir()
            (module / "source.py").write_text("replacement = True\n")
        with pytest.raises(RuntimeError, match="submodule changes"):
            await lease.diff()
        assert candidate.is_dir()
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
    finally:
        await lease.cleanup()
        await environment.cleanup()


async def test_colon_prefixed_filename_is_not_raw_diff_metadata(tmp_path):
    source = _repository(tmp_path)
    name = ":160000 160000 file.txt"
    (source / name).write_text("original\n")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "record colon filename")
    environment = LocalEnvironment(str(source))
    backend = EnvCandidateWorkspace(environment)
    lease = await backend.acquire("colon-filename")
    try:
        await lease.environment.write_file(name, "changed\n")
        patch = await lease.diff()
        await backend.adopt(patch)
        assert (source / name).read_text() == "changed\n"
    finally:
        await lease.cleanup()
        await environment.cleanup()


async def test_public_sdk_reports_capture_failure_and_retains_submodule_result(tmp_path):
    source, module_path = _source_with_dependency(tmp_path, False)
    environment = LocalEnvironment(str(source))
    backend = EnvCandidateWorkspace(environment)
    leases = []
    acquire = backend.acquire

    async def capture_lease(label):
        lease = await acquire(label)
        leases.append(lease)
        return lease

    backend.acquire = capture_lease

    async def repair(child, _inputs):
        root = Path(child.workspace_root)
        (root / "source.py").write_text("value = 2\n")
        (root / module_path / "source.py").write_text("value = 2\n")
        return "dependency checked"

    async def submit(ctx, _inputs):
        candidate = await ctx.candidate_workflow(repair, {}, label="dependency")
        await ctx.adopt_candidate(candidate)
        return "done"

    try:
        result = await OpenCollab(source, environment=environment).workflow(
            submit, candidate_workspace=backend, trace=False,
        )
        assert result.status == "failed"
        assert isinstance(result.error, CandidateCaptureError)
        assert result.output is None
        candidate = Path(leases[0].candidate_workspace)
        assert (candidate / "source.py").read_text() == "value = 2\n"
        assert (candidate / module_path / "source.py").read_text() == "value = 2\n"
        assert (source / "source.py").read_text() == "value = 1\n"
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
    finally:
        for lease in leases:
            await lease.cleanup()
        await environment.cleanup()
