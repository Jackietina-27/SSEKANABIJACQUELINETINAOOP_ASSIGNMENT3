"""Errors raised by the weather side of this project.

Grouped under one base so a caller can catch everything, with specific types
so it can also catch exactly one thing. Nothing here returns None or a
sentinel to signal failure.
"""
from __future__ import annotations

__all__ = ["AmbiguousName", "MalformedPayload", "RejectedRequest", "TemporaryFailure",
           "Unreachable", "UnknownPlace", "WeatherFailure"]


class WeatherFailure(Exception):
    """Base class for every weather error raised here."""


class TemporaryFailure(WeatherFailure):
    """One attempt failed in a way that may clear on a retry."""


class Unreachable(WeatherFailure):
    """Every attempt failed."""

    def __init__(self, url: str, attempts: int,
                 cause: Exception | None = None) -> None:
        self.url, self.attempts, self.cause = url, attempts, cause
        trailer = f" ({type(cause).__name__}: {cause})" if cause else ""
        super().__init__(f"{url} unreachable after {attempts} attempt(s){trailer}")


class RejectedRequest(WeatherFailure):
    """The provider rejected the request itself; retrying cannot help."""

    def __init__(self, url: str, status: int, reason: str | None = None) -> None:
        self.url, self.status, self.reason = url, status, reason
        super().__init__(f"{url} -> {status}" + (f": {reason}" if reason else ""))


class UnknownPlace(WeatherFailure):
    """Geocoding returned no candidates.

    Worth a named type because the API signals it by leaving results out of an
    otherwise ordinary 200 response, which is easy to mistake for a parser bug
    when it is documented behaviour of the service.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"no place matched {name!r}")


class AmbiguousName(WeatherFailure):
    """Several places matched and no choice was made."""

    def __init__(self, name: str, candidates: list) -> None:
        self.name, self.candidates = name, candidates
        super().__init__(f"{name!r} matched {len(candidates)} places; "
                         f"a choice is required")


class MalformedPayload(WeatherFailure):
    """Valid JSON arrived, but without the fields it was supposed to carry."""
