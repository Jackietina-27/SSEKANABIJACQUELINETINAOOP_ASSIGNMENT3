"""The providers, behind a structural protocol instead of a base class.

ForecastProvider is a typing.Protocol. Open-Meteo and MET Norway satisfy it
because they have the right methods, not because they inherit anything. The
two share no implementation whatever: one reads a daily block straight out of
the payload, the other rebuilds daily figures from an hourly series. A base
class would have been an empty one.

geocode() and get_forecast() are the functions the brief names. Both take a
caller, something that turns a RequestPlan into a Reply, defaulting to the
recorded one so importing this module cannot produce a surprise request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import pandas as pd

from src.common.errors import RejectedRequest, Unreachable
from src.common.request import BackoffSchedule, Reply, RequestPlan, execute
from src.weather.parsing import (COLUMNS, DAILY_VARIABLES, Location,
                                 locations_in, metno_to_daily, open_meteo_to_daily)

__all__ = ["FORECAST", "GEOCODE", "INDEX", "METNO", "MetNorway", "OpenMeteo",
           "PROVIDERS", "RECORDINGS", "Recorder", "ForecastProvider",
           "available_recordings", "fetch", "geocode", "get_forecast",
           "live_caller", "recorded_caller", "user_agent"]

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
METNO = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

RECORDINGS = Path(__file__).resolve().parents[2] / "data" / "samples"

# Which saved reply answers which request.
INDEX: dict[tuple[str, str], str] = {
    ("geocode", "dresden"): "geocode_dresden",
    ("geocode", "kampala"): "geocode_kampala",
    ("geocode", "nairobi"): "geocode_nairobi",
    ("geocode", "mbarara"): "geocode_mbarara",
    ("geocode", "entebbe"): "geocode_entebbe",
    ("geocode", "springfield"): "geocode_springfield_ambiguous",
    ("geocode", "zzzqqxyznowhere"): "geocode_zero_results",
    ("forecast", "0.32,32.58"): "forecast_kampala",
    ("forecast", "-0.60,30.65"): "forecast_mbarara",
    ("forecast", "0.06,32.48"): "forecast_entebbe",
    ("forecast", "51.05,13.74"): "forecast_dresden",
    ("forecast", "-1.28,36.82"): "forecast_nairobi",
    ("forecast", "invalid"): "forecast_invalid_latitude",
    ("metno", "0.32,32.58"): "metno_kampala",
    ("metno", "-0.60,30.65"): "metno_mbarara",
    ("metno", "0.06,32.48"): "metno_entebbe",
}


def user_agent(contact: str) -> str:
    """The User-Agent MET Norway requires, built from a caller's contact."""
    if not contact or not str(contact).strip():
        raise ValueError("a contact string is required")
    return f"advent2026-assignment2/1.0 ({str(contact).strip()})"


# ----------------------------------------------------------------------
# Callers: a plan goes in, a reply comes out
# ----------------------------------------------------------------------
def live_caller(contact: str, schedule: BackoffSchedule | None = None,
                session=None) -> Callable[[RequestPlan], Reply]:
    """A caller that performs real requests."""
    agent = user_agent(contact)
    plan_schedule = schedule or BackoffSchedule()

    def call(plan: RequestPlan) -> Reply:
        return execute(plan.with_headers(**{"User-Agent": agent,
                                            "Accept": "application/json"}),
                       plan_schedule, session)
    call.live = True
    call.schedule = plan_schedule
    return call


class Recorder:
    """A caller that replays saved replies. Never touches the network.

    Not a subclass of anything: it shares an interface with the live caller and
    no implementation at all, so inheritance would add coupling without
    removing a line of duplication.
    """

    live = False

    def __init__(self, directory: Path = RECORDINGS,
                 index: Mapping[tuple[str, str], str] | None = None) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"no recordings at {self.directory}")
        self.index = dict(index or INDEX)

    def __repr__(self) -> str:
        return f"Recorder({len(self.index)} saved replies)"

    def __len__(self) -> int:
        return len(self.index)

    def load(self, stem: str) -> Reply:
        """Read one saved reply by file stem."""
        path = self.directory / f"{stem}.json"
        if not path.exists():
            raise Unreachable(stem, 0, FileNotFoundError(path.name))
        saved = json.loads(path.read_text(encoding="utf-8"))
        meta, body = saved["_recorded"], saved["body"]
        if 400 <= meta["status"] < 500:
            raise RejectedRequest(meta["url"], meta["status"],
                                  body.get("reason") if isinstance(body, dict)
                                  else None)
        return Reply(meta["url"], meta["status"], meta["content_type"],
                     meta["elapsed_ms"], body, attempts=1, recorded=True)

    def __call__(self, plan: RequestPlan) -> Reply:
        stem = self._match(plan)
        if stem is None:
            raise Unreachable(plan.url, 0,
                              KeyError("no saved reply matches this plan"))
        return self.load(stem)

    def _match(self, plan: RequestPlan) -> str | None:
        params = plan.params
        if "geocoding-api" in plan.url:
            return self.index.get(
                ("geocode", str(params.get("name", "")).strip().lower()))
        if "api.met.no" in plan.url:
            key = f"{float(params.get('lat', 0)):.2f},{float(params.get('lon', 0)):.2f}"
            return self.index.get(("metno", key))
        if "open-meteo" in plan.url:
            lat = float(params.get("latitude", 0))
            lon = float(params.get("longitude", 0))
            if not -90 <= lat <= 90:
                return self.index.get(("forecast", "invalid"))
            return self.index.get(("forecast", f"{lat:.2f},{lon:.2f}"))
        return None


def recorded_caller() -> Recorder:
    """The default caller: saved replies, no network."""
    return Recorder()


def available_recordings(directory: Path = RECORDINGS) -> list[str]:
    return sorted(path.stem for path in Path(directory).glob("*.json"))


# ----------------------------------------------------------------------
# The providers
# ----------------------------------------------------------------------
@runtime_checkable
class ForecastProvider(Protocol):
    """Anything that can plan a forecast request and read the reply.

    A protocol, not an ABC, because the two providers share no implementation.
    Satisfying this is a matter of having the right methods, which is what
    structural typing is for.
    """

    name: str
    licence: str

    def plan(self, latitude: float, longitude: float,
             days: int) -> RequestPlan: ...

    def read(self, reply: Reply, days: int) -> pd.DataFrame: ...


@dataclass(frozen=True)
class OpenMeteo:
    """The main source. Publishes daily aggregates, so reading is direct."""

    name: str = "Open-Meteo"
    licence: str = "Weather data by Open-Meteo.com (CC BY 4.0)"

    def plan(self, latitude: float, longitude: float, days: int) -> RequestPlan:
        if not 1 <= int(days) <= 16:
            raise ValueError("Open-Meteo serves between 1 and 16 forecast days")
        return RequestPlan(FORECAST, {
            "latitude": latitude, "longitude": longitude,
            "daily": ",".join(DAILY_VARIABLES), "timezone": "auto",
            "forecast_days": int(days)}, label="open-meteo forecast")

    def read(self, reply: Reply, days: int) -> pd.DataFrame:
        return open_meteo_to_daily(reply.body).head(int(days))


@dataclass(frozen=True)
class MetNorway:
    """The second source. Publishes hourly data, so reading rolls it up.

    Coordinates are rounded to two decimals, about a kilometre. That is coarser
    than the provider's documented convention and finer than any forecast model
    resolves, so it costs nothing and sends materially less about where the
    user is.
    """

    name: str = "MET Norway"
    licence: str = ("Data from MET Norway Locationforecast 2.0, "
                    "CC BY 4.0 / NLOD")
    precision: int = 2

    def plan(self, latitude: float, longitude: float, days: int) -> RequestPlan:
        return RequestPlan(METNO, {"lat": round(float(latitude), self.precision),
                                   "lon": round(float(longitude), self.precision)},
                           label="met.no locationforecast")

    def read(self, reply: Reply, days: int) -> pd.DataFrame:
        return metno_to_daily(reply.body, int(days))


# The providers, by name.
PROVIDERS: dict[str, ForecastProvider] = {
    "Open-Meteo": OpenMeteo(),
    "MET Norway": MetNorway(),
}


def fetch(provider: ForecastProvider, caller, latitude: float, longitude: float,
          days: int = 7) -> tuple[pd.DataFrame, Reply]:
    """Plan, call, read: the same three steps for every provider."""
    plan = provider.plan(latitude, longitude, days)
    reply = caller(plan)
    return provider.read(reply, days), reply


# ----------------------------------------------------------------------
# The functions the brief names
# ----------------------------------------------------------------------
def geocode(name: str, caller=None, count: int = 5) -> list[Location]:
    """Find candidate locations for a place name."""
    if not name or not str(name).strip():
        raise ValueError("a place name is required")
    caller = caller or recorded_caller()
    plan = RequestPlan(GEOCODE, {"name": str(name).strip(), "count": int(count),
                                 "language": "en", "format": "json"},
                       label="geocoding")
    return locations_in(caller(plan).body, name)


def get_forecast(lat: float, lon: float, days: int = 7, caller=None) -> pd.DataFrame:
    """Daily forecast for one point, as a table with units in the column names.

    Coordinates are checked before a plan is even built, so an obvious mistake
    costs no request and the message names the real problem.
    """
    if not -90 <= float(lat) <= 90:
        raise ValueError(f"latitude {lat} is outside -90..90")
    if not -180 <= float(lon) <= 180:
        raise ValueError(f"longitude {lon} is outside -180..180")
    caller = caller or recorded_caller()
    table, _ = fetch(PROVIDERS["Open-Meteo"], caller, lat, lon, days)
    return table
