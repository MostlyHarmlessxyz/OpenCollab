"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { openBlueprint } = require("./harness.cjs");

function team(existing = "coder", adHoc = "") {
  return `entry: lead
roles:
  lead:
    prompt: Coordinate the task.
    tools: [message_agent]
  ${JSON.stringify(existing)}:
    prompt: Implement the task.
    tools: [file_read]
  auditor:
    prompt: Review the task.
    tools: [file_read]
topology:
  lead: [${JSON.stringify(existing)}, auditor${adHoc ? ", " + adHoc : ""}]
`;
}

for (const name of ["coder", "CODER", "CoDeR"]) {
  test(`renaming another role to ${name} preserves the existing identity`, t => {
    const view = openBlueprint(t, team());
    const before = view.resolvedConfig(view.source);
    view.click('[data-act="edit"]');
    view.input('[data-rename="auditor"]', name, "change");
    assert.deepEqual(Object.keys(view.exported().roles), ["lead", "coder", "auditor"]);
    assert.deepEqual(view.resolvedConfig(view.saveExport()), before);
  });
}

for (const name of ["inspector", "AUDITOR"]) {
  test(`distinct or self case-only rename to ${name} remains valid`, t => {
    const view = openBlueprint(t, team());
    view.resolvedConfig(view.source);
    view.click('[data-act="edit"]');
    view.input('[data-rename="auditor"]', name, "change");
    assert.deepEqual(Object.keys(view.exported().roles), ["lead", "coder", name]);
    assert.equal(view.document.querySelectorAll(".issue.error").length, 0);
    assert.deepEqual(view.resolvedConfig(view.saveExport()).topology.lead.sort(), ["coder", name].sort());
  });
}

test("ASCII collision checks preserve distinct non-ASCII identities", t => {
  const view = openBlueprint(t, team("kimi"));
  view.click('[data-act="edit"]');
  view.input('[data-rename="auditor"]', "kımi", "change");
  assert.ok(view.exported().roles.kımi);
  assert.ok(view.resolvedConfig(view.saveExport()).roles.kımi);
});

test("case variants of referenced ad-hoc roles cannot merge topology nodes", t => {
  const view = openBlueprint(t, team("coder", "ghost"));
  const before = view.resolvedConfig(view.source);
  view.click('[data-act="edit"]');
  view.input('[data-rename="auditor"]', "GHOST", "change");
  assert.deepEqual([...view.exported().topology.lead], ["coder", "auditor", "ghost"]);
  assert.deepEqual(view.resolvedConfig(view.saveExport()), before);
});

test("loaded case-colliding roles are reported until one identity is repaired", t => {
  const view = openBlueprint(t, team().replaceAll("auditor", "CODER"));
  assert.throws(
    () => view.resolvedConfig(view.source),
    error => /role identities collide/.test(String(error.stderr)),
  );
  const errors = [...view.document.querySelectorAll(".issue.error")];
  assert.ok(errors.some(error => /coder/i.test(error.textContent) && /identity/i.test(error.textContent)));
  assert.equal(view.document.querySelector(".stats .pill.ok"), null);
  view.click('[data-act="edit"]');
  view.input('[data-rename="CODER"]', "auditor", "change");
  assert.equal(view.document.querySelectorAll(".issue.error").length, 0);
  assert.ok(view.resolvedConfig(view.saveExport()).roles.auditor);
});

test("new role names skip existing ASCII case variants", t => {
  const view = openBlueprint(t, team("ROLE"));
  view.resolvedConfig(view.source);
  view.click('[data-act="edit"]');
  view.click('[data-act="add-role"]');
  assert.ok(view.exported().roles.role2);
  view.input('[data-prompt="role2"]', "Inspect the task.");
  view.click('[data-tool-role="role2"][data-tool="file_read"]');
  view.change('[data-edge-src="lead"][data-edge-dst="role2"]', true);
  assert.equal(view.document.querySelectorAll(".issue.error").length, 0);
  assert.ok(view.resolvedConfig(view.saveExport()).roles.role2);
});
