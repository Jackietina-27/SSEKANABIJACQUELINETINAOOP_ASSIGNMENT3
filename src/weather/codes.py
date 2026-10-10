"""WMO 4677 weather codes, as published by Open-Meteo.

A mapping from code to a (description, precipitating) pair, so "what is it
called" and "does rain reach the ground" come from one table and cannot fall
out of step.
"""
from __future__ import annotations

from typing import Iterator

__all__ = ["CODES", "FAMILIES", "explain", "known", "precipitating"]

# code -> (description, does precipitation reach the ground)
CODES: dict[int, tuple[str, bool]] = {
    0: ("Clear sky", False),
    1: ("Mainly clear", False),
    2: ("Partly cloudy", False),
    3: ("Overcast", False),
    45: ("Fog", False),
    48: ("Depositing rime fog", False),
    51: ("Light drizzle", True),
    53: ("Moderate drizzle", True),
    55: ("Dense drizzle", True),
    56: ("Light freezing drizzle", True),
    57: ("Dense freezing drizzle", True),
    61: ("Slight rain", True),
    63: ("Moderate rain", True),
    65: ("Heavy rain", True),
    66: ("Light freezing rain", True),
    67: ("Heavy freezing rain", True),
    71: ("Slight snowfall", True),
    73: ("Moderate snowfall", True),
    75: ("Heavy snowfall", True),
    77: ("Snow grains", True),
    80: ("Slight rain showers", True),
    81: ("Moderate rain showers", True),
    82: ("Violent rain showers", True),
    85: ("Slight snow showers", True),
    86: ("Heavy snow showers", True),
    95: ("Thunderstorm", True),
    96: ("Thunderstorm with slight hail", True),
    99: ("Thunderstorm with heavy hail", True),
}

# Coarse families, used only for a code outside the published set.
FAMILIES: tuple[tuple[range, str], ...] = (
    (range(0, 4), "Cloud cover"), (range(45, 49), "Fog"),
    (range(51, 58), "Drizzle"), (range(61, 68), "Rain"),
    (range(71, 78), "Snow"), (range(80, 83), "Rain showers"),
    (range(85, 87), "Snow showers"), (range(95, 100), "Thunderstorm"),
)


def known(code: int | float | None) -> bool:
    """Whether a code appears in the published table."""
    try:
        return code is not None and int(code) in CODES
    except (TypeError, ValueError):
        return False


def explain(code: int | float | None) -> str:
    """A readable description of a weather code.

    An unpublished code falls back to its family, then to a labelled unknown.

    >>> explain(95)
    'Thunderstorm'
    >>> explain(None)
    'Not reported'
    """
    if code is None:
        return "Not reported"
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "Not reported"
    if value in CODES:
        return CODES[value][0]
    for span, family in FAMILIES:
        if value in span:
            return f"{family} (code {value})"
    return f"Unknown (code {value})"


def precipitating(code: int | float | None) -> bool:
    """Whether precipitation reaches the ground under this code.

    Not the same question as "is it cloudy": overcast is not wet.

    >>> precipitating(61), precipitating(3)
    (True, False)
    """
    try:
        return CODES[int(code)][1] if code is not None else False
    except (TypeError, ValueError, KeyError):
        return False
