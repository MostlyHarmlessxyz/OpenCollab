"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { openBlueprint } = require("./harness.cjs");

const source = `entry: lead
roles:
  lead:
    prompt: Plan the task.
    tools: [file_read, spawn_agent]
  coder:
    prompt: Implement the task.
    tools: [bash]
topology:
  lead: [coder]
`;

test("feedback survives edit mode and tool changes and reaches the clipboard", t => {
  const blueprint = openBlueprint(t, source);
  const feedback = 'Keep my feedback.\nReview <code> and "quoted text".\n</textarea><div id="injected">';
  const originalNotes = blueprint.document.querySelector('[data-out="notes"]');
  blueprint.input('[data-out="notes"]', feedback);

  blueprint.click('[data-act="edit"]');

  assert.notEqual(blueprint.document.querySelector('[data-out="notes"]'), originalNotes);
  assert.equal(blueprint.document.querySelector('[data-out="notes"]').value, feedback);
  blueprint.input('[data-model="lead"]', "edited-model");
  blueprint.click('[data-tool-role="lead"][data-tool="bash"]');
  assert.equal(blueprint.document.querySelector('[data-out="notes"]').value, feedback);
  assert.equal(blueprint.document.querySelector("#injected"), null);
  blueprint.click('[data-act="copy-feedback"]');
  assert.equal(blueprint.clipboard.length, 1);
  assert.ok(blueprint.clipboard[0].includes(`Notes:\n${feedback}\n`));
  assert.ok(blueprint.clipboard[0].includes("model: edited-model"));
});

test("feedback follows topology, rename, add, remove and entry edits", t => {
  const blueprint = openBlueprint(t, source);
  blueprint.input('[data-out="notes"]', "First feedback");
  blueprint.change('[data-edge-src="lead"][data-edge-dst="coder"]', false);
  assert.equal(blueprint.document.querySelector('[data-out="notes"]').value, "First feedback");
  blueprint.click('[data-act="edit"]');
  blueprint.input('[data-out="notes"]', "Revised feedback");
  blueprint.input('[data-rename="coder"]', "worker", "change");
  blueprint.change('[data-entry="worker"]', true);
  blueprint.click('[data-act="add-role"]');
  blueprint.click('[data-remove="role"]');

  assert.equal(blueprint.document.querySelector('[data-out="notes"]').value, "Revised feedback");
  assert.equal(blueprint.exported().entry, "worker");
  assert.equal(Object.hasOwn(blueprint.exported().roles, "coder"), false);
  assert.equal(Object.hasOwn(blueprint.exported().roles, "role"), false);
  blueprint.input('[data-out="notes"]', "");
  blueprint.click('[data-act="edit"]');
  assert.equal(blueprint.document.querySelector('[data-out="notes"]').value, "");
  blueprint.click('[data-act="copy-feedback"]');
  assert.ok(blueprint.clipboard[0].includes("Notes:\n(none)\n"));
});
