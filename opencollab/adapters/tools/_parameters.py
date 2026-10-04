"""Normalize JSON tool parameters for operations requiring Python integers."""

import math


def integer_parameter(value: object, name: str) -> int:
    """Accept the mathematical integers supported by the tool schemas."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"{name} must be an integer")
