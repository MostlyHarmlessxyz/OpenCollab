"""Public team metadata reports the entry seat before the remaining roles."""

from opencollab.teams import (
    declared_role_names,
    declared_role_profiles,
    declared_role_prompt_digests,
    declared_role_tools,
)


def test_public_team_helpers_put_entry_first_with_consistent_remaining_order(tmp_path):
    path = tmp_path / "team.yaml"
    path.write_text(
        "entry: lead\nroles:\n"
        "  coder:\n    prompt: code\n    tools: [file_read]\n"
        "  lead:\n    prompt: lead\n    profile: single2\n    tools: [grep]\n"
        "  tester:\n    prompt: test\n    tools: [git_diff]\n",
        encoding="utf-8",
    )
    expected = ("lead", "coder", "tester")

    assert declared_role_names(str(path)) == expected
    for helper in (declared_role_tools, declared_role_prompt_digests, declared_role_profiles):
        assert tuple(helper(str(path))) == expected
    assert declared_role_tools(str(path))["lead"] == ("grep",)
    assert declared_role_profiles(str(path))["lead"] == "single2"
