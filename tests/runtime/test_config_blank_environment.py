"""Blank environment values allow usable env-file defaults to resolve."""

import os

import pytest

from opencollab.bootstrap.config import build_config


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
@pytest.mark.parametrize(("variable", "field", "value", "expected"), [
    ("OPENCOLLAB_API_KEY", "api_key", "fixture-key", "fixture-key"),
    ("OPENCOLLAB_MODEL", "model", "file-model", "file-model"),
    ("OPENCOLLAB_PROVIDER", "provider", "openai", "openai"),
    ("OPENCOLLAB_BASE_URL", "base_url", "https://fixture.invalid/v1", "https://fixture.invalid/v1"),
    ("OPENCOLLAB_WIRE_PROTOCOL", "wire_protocol", "responses", "responses"),
    ("OPENCOLLAB_BUDGET", "budget", "42", 42),
    ("OPENCOLLAB_THINKING", "thinking", "true", True),
    ("OPENCOLLAB_LLM_TIMEOUT", "llm_timeout", "12", 12.0),
])
def test_blank_environment_value_uses_the_same_file_variable(
    tmp_path, monkeypatch, blank, variable, field, value, expected,
):
    for name in tuple(os.environ):
        if name.startswith("OPENCOLLAB_") or name in {
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL",
        }:
            monkeypatch.delenv(name)
    path = tmp_path / "runtime.env"
    path.write_text(f"{variable}={value}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCOLLAB_CONFIG_FILE", str(path))
    monkeypatch.setenv(variable, blank)

    assert getattr(build_config(str(tmp_path)), field) == expected
