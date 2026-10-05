"""Usage projection for the OpenAI Responses transport."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.llm.types import (
    Usage,
    estimate_messages_tokens,
    estimate_tokens,
    usage_to_dict,
)


def _optional_usage_int(source: Any, key: str) -> int | None:
    if not isinstance(source, dict) or source.get(key) is None:
        return None
    if isinstance(source[key], bool):
        return None
    try:
        value = int(source[key])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _positive_usage_int(source: Any, key: str) -> int | None:
    value = _optional_usage_int(source, key)
    return value if value is not None and value > 0 else None


def _reported_responses_usage(response: Any) -> Usage | None:
    """Retain only counters actually returned by a failed attempt."""
    raw = usage_to_dict(getattr(response, "usage", None))
    input_tokens = _optional_usage_int(raw, "input_tokens")
    output_tokens = _optional_usage_int(raw, "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    cache_creation = _optional_usage_int(input_details, "cache_write_tokens")
    if cache_creation is None:
        cache_creation = _optional_usage_int(raw, "cache_write_tokens")
    return Usage(
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        cache_read_tokens=_optional_usage_int(input_details, "cached_tokens"),
        cache_creation_tokens=cache_creation,
        reasoning_tokens=_optional_usage_int(output_details, "reasoning_tokens"),
        estimated=input_tokens is None or output_tokens is None,
        raw_usage=raw,
    )


def _combine_responses_usage(attempts: list[Usage | None]) -> Usage | None:
    """Sum known attempts and retain omissions in the native usage record."""
    known = [usage for usage in attempts if usage is not None]
    if not known:
        return None
    if len(attempts) == 1:
        return known[0]

    def optional_total(name: str) -> int | None:
        values = [getattr(usage, name) if usage is not None else None for usage in attempts]
        return None if any(value is None for value in values) else sum(values)

    return Usage(
        input_tokens=sum(usage.input_tokens for usage in known),
        output_tokens=sum(usage.output_tokens for usage in known),
        cache_read_tokens=optional_total("cache_read_tokens"),
        cache_creation_tokens=optional_total("cache_creation_tokens"),
        reasoning_tokens=optional_total("reasoning_tokens"),
        estimated=any(usage is None or usage.estimated for usage in attempts),
        raw_usage={"attempts": [usage.raw_usage if usage is not None else None for usage in attempts]},
        context_tokens=known[-1].context_tokens if known[-1].context_tokens is not None else known[-1].input_tokens,
    )


def parse_responses_usage(
    response: Any,
    messages: list[dict[str, Any]],
    content: str | None,
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> Usage:
    """Build normalized usage while retaining provider-native counters.

    ``input_tokens`` is estimated when a Responses endpoint omits its usage
    counters.  The request's registered tool schemas are part of that input,
    so callers must pass the provider-shaped tool list through to the
    estimator as well as the conversational messages.
    """
    raw = usage_to_dict(getattr(response, "usage", None))
    input_tokens = _positive_usage_int(raw, "input_tokens")
    output_tokens = _positive_usage_int(raw, "output_tokens")
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    cache_creation_tokens = _optional_usage_int(input_details, "cache_write_tokens")
    if cache_creation_tokens is None:
        cache_creation_tokens = _optional_usage_int(raw, "cache_write_tokens")
    estimated = input_tokens is None or output_tokens is None
    if input_tokens is None:
        input_tokens = estimate_messages_tokens(messages, tools)
    if output_tokens is None:
        text = content or ""
        for call in tool_calls:
            function = call["function"]
            text += str(function.get("name") or "") + str(function.get("arguments") or "")
        output_tokens = estimate_tokens(text) if text else 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=_optional_usage_int(input_details, "cached_tokens"),
        cache_creation_tokens=cache_creation_tokens,
        reasoning_tokens=_optional_usage_int(output_details, "reasoning_tokens"),
        estimated=estimated,
        raw_usage=raw,
    )


__all__ = ["parse_responses_usage"]
