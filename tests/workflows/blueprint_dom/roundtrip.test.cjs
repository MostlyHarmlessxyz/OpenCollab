"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { openBlueprint } = require("./harness.cjs");

const roles = `entry: lead
roles:
  lead:
    prompt: Plan the task.
    model: fixture-model
    temperature: 0.4
    thinking: false
    thinking_params: {type: disabled}
    profile: single2
    tools: [file_read, spawn_agent]
  reviewer:
    prompt_file: review.md
    tools: [git_diff]
topology:
  lead: [reviewer]
tool_limits:
  file_read: {max_read_chars: 5000}
hooks:
  Notification:
    - command: echo fixture
`;

for (const context of [
  "context: no_history_compaction\n",
  "context: {policy: no_history_compaction, tool_result_budget: 20000}\n",
  "context: {tool_result_budget: 12000}\n",
]) {
  test(`supported role and context settings survive export with ${context.trim()}`, t => {
    const blueprint = openBlueprint(t, roles + context);
    const exported = blueprint.exported();
    assert.deepEqual(exported.roles.lead, blueprint.original.roles.lead);
    assert.deepEqual(exported.roles.reviewer, blueprint.original.roles.reviewer);
    assert.deepEqual(exported.context, blueprint.original.context);
    assert.deepEqual(exported.topology, blueprint.original.topology);
    assert.deepEqual(exported.tool_limits, blueprint.original.tool_limits);
    assert.deepEqual(exported.hooks, blueprint.original.hooks);
    assert.deepEqual(blueprint.resolvedConfig(blueprint.saveExport()), blueprint.resolvedConfig(blueprint.source));

    blueprint.click('[data-act="edit"]');
    blueprint.input('[data-model="lead"]', "edited-model");
    blueprint.click('[data-tool-role="lead"][data-tool="bash"]');
    const edited = blueprint.exported();
    assert.equal(edited.roles.lead.model, "edited-model");
    assert.ok(edited.roles.lead.tools.includes("bash"));
    assert.equal(edited.roles.lead.profile, "single2");
    assert.deepEqual(edited.context, blueprint.original.context);
    assert.equal(edited.roles.lead.thinking, false);
    assert.deepEqual(edited.roles.lead.thinking_params, blueprint.original.roles.lead.thinking_params);
    assert.equal(blueprint.resolvedConfig(blueprint.saveExport()).roles.lead.profile, "single2");
  });
}

test("omitted profile and context keep their loader defaults", t => {
  const blueprint = openBlueprint(t, "roles:\n  lead:\n    prompt: Solve the task.\n    tools: [file_read]\n");
  const exported = blueprint.exported();
  assert.equal(Object.hasOwn(exported, "context"), false);
  assert.equal(Object.hasOwn(exported.roles.lead, "profile"), false);
  assert.deepEqual(blueprint.resolvedConfig(blueprint.saveExport()), blueprint.resolvedConfig(blueprint.source));
});
