"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { openBlueprint } = require("./harness.cjs");

const cases = [
  {
    name: "roles inherit the team default",
    teamBudget: "budget: {tokens: 20000}\n",
    budgets: { lead: 20000, reviewer: 20000 },
  },
  {
    name: "a role override wins while null inherits the team default",
    teamBudget: "budget: {tokens: 20000}\n",
    leadBudget: "    budget: null\n",
    reviewerBudget: "    budget: {tokens: 30000}\n",
    budgets: { lead: 20000, reviewer: 30000 },
  },
  {
    name: "each role declares an allowance with the team budget omitted",
    leadBudget: "    budget: {tokens: 10000}\n",
    reviewerBudget: "    budget: {tokens: 30000}\n",
    budgets: { lead: 10000, reviewer: 30000 },
  },
  {
    name: "each role declares an allowance with a null team budget",
    teamBudget: "budget: null\n",
    leadBudget: "    budget: {tokens: 10000}\n",
    reviewerBudget: "    budget: {tokens: 30000}\n",
    budgets: { lead: 10000, reviewer: 30000 },
  },
  { name: "omitted allowances retain the shared budget", budgets: {} },
  {
    name: "null and omitted allowances retain the shared budget",
    teamBudget: "budget: null\n",
    leadBudget: "    budget: null\n",
    budgets: {},
  },
];

for (const fixture of cases) {
  test(`token budgets survive export when ${fixture.name}`, t => {
    const source = (fixture.teamBudget || "") + `entry: lead
roles:
  lead:
    prompt: Plan the task.
    tools: [file_read, spawn_agent]
${fixture.leadBudget || ""}  reviewer:
    prompt: Review the patch.
    tools: [git_diff]
${fixture.reviewerBudget || ""}topology:
  lead: [reviewer]
`;
    const blueprint = openBlueprint(t, source);
    const original = blueprint.resolvedConfig(blueprint.source);
    assert.deepEqual(original.role_budgets, fixture.budgets);
    assert.deepEqual(blueprint.resolvedConfig(blueprint.saveExport()), original);
    const exported = blueprint.exported();
    if (blueprint.original.budget != null) assert.deepEqual(exported.budget, blueprint.original.budget);
    for (const name of Object.keys(blueprint.original.roles)) {
      const budget = blueprint.original.roles[name].budget;
      if (budget != null) assert.deepEqual(exported.roles[name].budget, budget);
    }

    blueprint.click('[data-act="edit"]');
    blueprint.input('[data-model="lead"]', "edited-model");
    blueprint.click('[data-tool-role="lead"][data-tool="bash"]');
    const edited = blueprint.resolvedConfig(blueprint.saveExport());
    assert.deepEqual(edited.role_budgets, fixture.budgets);
    assert.equal(edited.roles.lead.model, "edited-model");
    assert.ok(edited.roles.lead.tools.includes("bash"));
  });
}
