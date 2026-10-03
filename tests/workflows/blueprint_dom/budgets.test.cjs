"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
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

function sourceFor(fixture) {
  return (fixture.teamBudget || "") + `entry: lead
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
}

for (const fixture of cases) {
  test(`token budgets survive export when ${fixture.name}`, t => {
    const source = sourceFor(fixture);
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

  test(`adding a role checks budget coverage when ${fixture.name}`, t => {
    const blueprint = openBlueprint(t, sourceFor(fixture));
    assert.deepEqual(blueprint.resolvedConfig(blueprint.source).role_budgets, fixture.budgets);
    blueprint.click('[data-act="edit"]');
    blueprint.click('[data-act="add-role"]');
    blueprint.input('[data-prompt="role"]', "Read the requested file.");
    blueprint.click('[data-tool-role="role"][data-tool="file_read"]');
    blueprint.change('[data-edge-src="lead"][data-edge-dst="role"]', true);

    const exported = blueprint.exported();
    for (const name of Object.keys(blueprint.original.roles)) {
      if (blueprint.original.roles[name].budget != null) {
        assert.deepEqual(exported.roles[name].budget, blueprint.original.roles[name].budget);
      }
    }
    const errors = [...blueprint.document.querySelectorAll(".issue.error")];
    const missingBudget = Object.keys(fixture.budgets).length > 0 && blueprint.original.budget == null;
    if (!missingBudget) {
      assert.equal(errors.length, 0);
      const expected = Object.keys(fixture.budgets).length
        ? { ...fixture.budgets, role: blueprint.original.budget.tokens } : {};
      assert.deepEqual(blueprint.resolvedConfig(blueprint.saveExport()).role_budgets, expected);
      return;
    }

    assert.throws(
      () => blueprint.resolvedConfig(blueprint.saveExport()),
      error => /roles \['role'\] have no token budget/.test(String(error.stderr)),
    );
    assert.equal(errors.length, 1);
    assert.equal(errors[0].querySelector(".who").textContent, "role");
    assert.match(errors[0].textContent, /budget\.tokens.*team.*role/);
    assert.equal(blueprint.document.querySelector(".stats .pill.ok"), null);

    blueprint.input('[data-out="notes"]', "Set a budget for the added role.");
    blueprint.click('[data-act="copy-feedback"]');
    assert.ok(blueprint.clipboard[0].includes("Set a budget for the added role."));
    assert.ok(blueprint.clipboard[0].includes("tokens: 10000"));

    const repairedPath = path.join(blueprint.directory, "repaired.yaml");
    fs.writeFileSync(repairedPath, JSON.stringify({ ...exported, budget: { tokens: 5000 } }));
    assert.deepEqual(blueprint.resolvedConfig(repairedPath).role_budgets, { ...fixture.budgets, role: 5000 });
    exported.roles.role.budget = { tokens: 7000 };
    fs.writeFileSync(repairedPath, JSON.stringify(exported));
    assert.deepEqual(blueprint.resolvedConfig(repairedPath).role_budgets, { ...fixture.budgets, role: 7000 });
  });
}
