"""A request described as a value, and a function that carries it out.

RequestPlan is a frozen description of what to call, with what parameters,
under what headers and timeout. Constructing one does nothing, so a plan can
be built, inspected, logged, compared and tested with no network in sight.
execute() is the only thing that turns a plan into traffic.

That split follows the brief's own requirement to report the status code,
content type and timing of a request. Those are properties of a result, and a
result is easier to reason about when the thing that produced it could be
inspected first.

Retrying is driven by BackoffSchedule, a generator of waits. Being a
generator and not a loop inside a client, the schedule can be listed, summed
and asserted without a second of real waiting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

import requests

from src.common.errors import RejectedRequest, TemporaryFailure, Unreachable

__all__ = ["BackoffSchedule", "RequestPlan", "Reply", "execute"]

# Statuses worth another attempt. Anything else in the 4xx range will not change.
RETRY_ON = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RequestPlan:
    """A request, described but not performed."""

    url: str
    params: Mapping[str, Any] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.url or not str(self.url).startswith("http"):
            raise ValueError(f"a plan needs an http(s) url, got {self.url!r}")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def with_headers(self, **extra: str) -> "RequestPlan":
        """A copy carrying additional headers; the plan itself stays frozen."""
        merged = {**dict(self.headers), **extra}
        return RequestPlan(self.url, self.params, merged, self.timeout, self.label)

    @property
    def cache_key(self) -> str:
        """A stable key for this plan, used to find a recorded reply."""
        parts = "&".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.url}?{parts}"

    def __str__(self) -> str:
        name = self.label or self.url.rsplit("/", 1)[-1]
        return f"{name} ({len(self.params)} params, timeout {self.timeout}s)"


@dataclass(frozen=True)
class Reply:
    """What came back, with the metadata Part A asks to be reported."""

    url: str
    status: int
    content_type: str
    elapsed_ms: float
    body: Mapping[str, Any]
    attempts: int = 1
    recorded: bool = False

    def __repr__(self) -> str:
        source = "recorded" if self.recorded else "live"
        return (f"Reply({self.status}, {self.elapsed_ms:.0f} ms, {source}, "
                f"attempts={self.attempts})")

    def details(self) -> dict[str, Any]:
        """Status, content type and timing, flattened for a table."""
        return {"status": self.status, "content_type": self.content_type,
                "elapsed_ms": round(self.elapsed_ms, 1),
                "attempts": self.attempts, "recorded": self.recorded,
                "url": self.url}


@dataclass(frozen=True)
class BackoffSchedule:
    """How long to wait between attempts, as an iterable of seconds."""

    attempts: int = 3
    start: float = 0.5
    multiplier: float = 2.0
    cap: float = 8.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.start < 0 or self.cap < 0:
            raise ValueError("waits cannot be negative")
        if self.multiplier < 1:
            raise ValueError("a multiplier below 1 would shrink the waits")
        if self.cap < self.start:
            raise ValueError("cap cannot be below start")

    def __iter__(self) -> Iterator[float]:
        """Yield the waits, one fewer than attempts.

        >>> list(BackoffSchedule(attempts=4, start=0.5))
        [0.5, 1.0, 2.0]
        """
        wait = self.start
        for _ in range(self.attempts - 1):
            yield min(wait, self.cap)
            wait *= self.multiplier

    def __len__(self) -> int:
        return max(self.attempts - 1, 0)

    def waits(self) -> list[float]:
        return list(self)

    def worst_case(self) -> float:
        """Longest possible time spent waiting, in seconds."""
        return sum(self)

    def __str__(self) -> str:
        return (f"{self.attempts} attempts, waits {self.waits()}, "
                f"at most {self.worst_case()}s waiting")


def execute(plan: RequestPlan, schedule: BackoffSchedule | None = None,
            session: requests.Session | None = None,
            sleep: Callable[[float], None] = time.sleep) -> Reply:
    """Carry out a plan, retrying transient failures.

    A RejectedRequest, meaning any 4xx that is not 408 or 429, is raised at
    once and never retried. An out-of-range latitude is just as invalid on the
    fourth attempt as the first, and repeating it only postpones the message
    the caller needs.
    """
    schedule = schedule or BackoffSchedule()
    caller = session or requests
    waits = schedule.waits()
    last: Exception | None = None

    for attempt in range(1, schedule.attempts + 1):
        started = time.perf_counter()
        try:
            response = caller.get(plan.url, params=dict(plan.params),
                                  headers=dict(plan.headers),
                                  timeout=plan.timeout)
            elapsed = (time.perf_counter() - started) * 1000

            if response.status_code in RETRY_ON:
                raise TemporaryFailure(
                    f"{response.status_code} {response.reason} from {plan.url}")
            if 400 <= response.status_code < 500:
                raise RejectedRequest(response.url, response.status_code,
                                      _reason(response))
            response.raise_for_status()
            return Reply(response.url, response.status_code,
                         response.headers.get("Content-Type", ""), elapsed,
                         response.json(), attempts=attempt)

        except RejectedRequest:
            raise
        except (TemporaryFailure, requests.RequestException, ValueError) as problem:
            last = problem
            if attempt < schedule.attempts:
                sleep(waits[attempt - 1])

    raise Unreachable(plan.url, schedule.attempts, last)


def _reason(response: requests.Response) -> str | None:
    """Open-Meteo puts a readable reason in its error bodies."""
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("reason") if isinstance(body, dict) else None
