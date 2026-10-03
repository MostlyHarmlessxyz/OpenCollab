"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const vm = require("node:vm");
const { parseHTML } = require("linkedom");

const repo = path.resolve(__dirname, "../../..");
const skill = path.join(repo, "skills/team-config");

function openBlueprint(t, yaml) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "opencollab-blueprint-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const source = path.join(directory, "team.yaml");
  const output = path.join(directory, "team.html");
  fs.writeFileSync(source, yaml);
  fs.writeFileSync(path.join(directory, "review.md"), "Review the patch.\n");
  execFileSync("sh", [path.join(skill, "build.sh"), source, output]);
  const { document, Event } = parseHTML(fs.readFileSync(output, "utf8"));
  const clipboard = [];
  const window = {
    document,
    navigator: { clipboard: { writeText: text => { clipboard.push(text); return Promise.resolve(); } } },
    setTimeout: () => 0,
    matchMedia: () => ({ matches: false }),
  };
  window.window = window;
  window.self = window;
  vm.createContext(window);
  for (const script of document.querySelectorAll("script")) {
    if (script.getAttribute("type") !== "text/yaml") vm.runInContext(script.textContent, window);
  }
  const fatal = document.getElementById("app").querySelector(".fatal");
  assert.ok(!fatal, fatal && fatal.textContent);

  return {
    document, clipboard, directory, source,
    original: window.jsyaml.load(document.querySelector("#team-src").textContent),
    exported: () => window.jsyaml.load(document.querySelector('[data-out="yaml"]').value),
    input(selector, value, type = "input") {
      const element = document.querySelector(selector);
      assert.ok(element, selector);
      element.value = value;
      element.dispatchEvent(new Event(type, { bubbles: true }));
    },
    click(selector) {
      const element = document.querySelector(selector);
      assert.ok(element, selector);
      element.dispatchEvent(new Event("click", { bubbles: true }));
    },
    change(selector, checked) {
      const element = document.querySelector(selector);
      assert.ok(element, selector);
      element.checked = checked;
      element.dispatchEvent(new Event("change", { bubbles: true }));
    },
    resolvedConfig(yamlPath) {
      const python = process.env.OPENCOLLAB_TEST_PYTHON || path.join(repo, ".venv/bin/python");
      return JSON.parse(execFileSync(python, ["-c", [
        "import json, sys",
        "from opencollab.bootstrap.team_config import load_team_config",
        "team = load_team_config(path=sys.argv[1])",
        "print(json.dumps({'roles': {name: role.model_dump() for name, role in team.roles.items()},",
        "'context': {'name': team.context.name, 'budget': team.context.tool_result_budget},",
        "'entry': team.entry, 'topology': {name: sorted(edges) for name, edges in team.topology.edges.items()},",
        "'tool_limits': team.tool_limits, 'role_budgets': team.role_budgets}))",
      ].join("\n"), yamlPath], { env: { ...process.env, PYTHONPATH: repo }, encoding: "utf8" }));
    },
    saveExport() {
      const exported = path.join(directory, "exported.yaml");
      fs.writeFileSync(exported, document.querySelector('[data-out="yaml"]').value);
      return exported;
    },
  };
}

module.exports = { openBlueprint };
