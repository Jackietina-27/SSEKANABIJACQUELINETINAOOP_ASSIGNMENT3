"""Pure functions from a decoded payload to a table.

Nothing here performs a request, imports requests or reads the clock. These
functions are where the real defects live, so they are kept somewhere they
can be exercised against saved JSON with no machinery at all.

The defect worth naming: the Open-Meteo geocoding endpoint reports "nothing
matched" by omitting the results key from an otherwise ordinary HTTP 200.
Writing payload["results"] therefore raises KeyError on a successful
response, and raises it wherever the payload is finally read instead of where
the request was made.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from src.common.errors import AmbiguousName, MalformedPayload, UnknownPlace
from src.weather.codes import explain

__all__ = ["COLUMNS", "DAILY_VARIABLES", "Location", "choose", "locations_in",
           "metno_to_daily", "open_meteo_to_daily"]

# The daily variables the brief requires.
DAILY_VARIABLES = ("weather_code", "temperature_2m_max", "temperature_2m_min",
                   "precipitation_sum", "wind_speed_10m_max")

# One column set for every provider, so two tables can be joined on date.
COLUMNS = ("date", "max_c", "min_c", "rain_mm", "wind_kmh", "code", "outlook")


@dataclass(frozen=True)
class Location:
    """A geocoded place. Frozen, because a name pinned to coordinates is a fact
    and not working state.
    """

    name: str
    country: str
    latitude: float
    longitude: float
    area: str = ""
    population: int | None = None
    timezone: str = ""

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude {self.latitude} is outside -90..90")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude {self.longitude} is outside -180..180")

    @property
    def described(self) -> str:
        return ", ".join([self.name] + [p for p in (self.area, self.country) if p])

    def rounded(self, places: int = 2) -> tuple[float, float]:
        """Coordinates at reduced precision.

        Two decimal places is about a kilometre: finer than any weather model
        resolves, and coarser than the four decimals a provider's documentation
        tends to ask for. Here so the privacy trade-off in Part D can be made
        and not only described.
        """
        return round(self.latitude, places), round(self.longitude, places)

    def __str__(self) -> str:
        crowd = f", pop. {self.population:,}" if self.population else ""
        return f"{self.described} ({self.latitude:.4f}, {self.longitude:.4f}){crowd}"


def locations_in(payload: Mapping[str, Any], name: str = "") -> list[Location]:
    """Read geocoding candidates from a payload.

    Uses .get("results"), never payload["results"]; the module docstring
    explains why that distinction is the point of this function.
    """
    if not isinstance(payload, Mapping):
        raise MalformedPayload("the geocoding payload was not an object")
    rows = payload.get("results")
    if not rows:
        raise UnknownPlace(name or "the requested name")

    found = []
    for row in rows:
        try:
            found.append(Location(
                name=row["name"], country=row.get("country", ""),
                latitude=float(row["latitude"]), longitude=float(row["longitude"]),
                area=row.get("admin1", ""), population=row.get("population"),
                timezone=row.get("timezone", "")))
        except (KeyError, TypeError, ValueError):
            continue        # one malformed candidate must not lose the others
    if not found:
        raise UnknownPlace(name or "the requested name")
    return found


def choose(candidates: Sequence[Location],
           preference: int | str | None = None) -> Location:
    """Resolve an ambiguous result to one place.

    A single candidate needs no choice. Several do, and if none is offered the
    candidates are raised so the caller can present them. Defaulting to the
    first match resolves an ambiguous name to whichever place the service
    happened to rank highest.
    """
    if not candidates:
        raise UnknownPlace("there were no candidates to choose from")
    if len(candidates) == 1:
        return candidates[0]
    if preference is None:
        raise AmbiguousName(candidates[0].name, list(candidates))
    if isinstance(preference, int):
        if not 0 <= preference < len(candidates):
            raise IndexError(
                f"preference {preference} is outside 0..{len(candidates) - 1}")
        return candidates[preference]
    wanted = str(preference).strip().lower()
    for place in candidates:
        if wanted in (place.country.lower(), place.area.lower()):
            return place
    raise UnknownPlace(f"no candidate in {preference!r}")


def open_meteo_to_daily(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Open-Meteo's daily block, as a tidy table with units in the names."""
    daily = payload.get("daily")
    if not isinstance(daily, Mapping) or "time" not in daily:
        raise MalformedPayload("the payload carries no daily block")
    missing = [name for name in DAILY_VARIABLES if name not in daily]
    if missing:
        raise MalformedPayload(f"the daily block is missing {missing}")

    frame = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "max_c": daily["temperature_2m_max"],
        "min_c": daily["temperature_2m_min"],
        "rain_mm": daily["precipitation_sum"],
        "wind_kmh": daily["wind_speed_10m_max"],
        "code": daily["weather_code"],
    })
    frame["outlook"] = [explain(code) for code in frame["code"]]
    return frame[list(COLUMNS)]


def metno_to_daily(payload: Mapping[str, Any],
                   days: int | None = None) -> pd.DataFrame:
    """Reduce MET Norway's hourly series to the same daily shape.

    MET Norway publishes instantaneous hourly values together with overlapping
    next_6_hours precipitation blocks, so the daily figures are built here: the
    maximum and minimum of the hourly air temperature, the maximum hourly wind,
    and the six-hourly rainfall sampled every sixth hour so the overlapping
    windows are not added twice.

    Wind arrives in metres per second and is converted to km/h. Without that
    the second provider contradicts the first by a factor of 3.6 and the
    cross-check is worthless.
    """
    series = (payload.get("properties") or {}).get("timeseries")
    if not series:
        raise MalformedPayload("the MET Norway payload has no timeseries")

    temperatures: dict[dt.date, list[float]] = defaultdict(list)
    winds: dict[dt.date, list[float]] = defaultdict(list)
    rainfall: dict[dt.date, float] = defaultdict(float)

    for entry in series:
        moment = dt.datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
        day = moment.date()
        instant = entry.get("data", {}).get("instant", {}).get("details", {})
        if "air_temperature" in instant:
            temperatures[day].append(float(instant["air_temperature"]))
        if "wind_speed" in instant:
            winds[day].append(float(instant["wind_speed"]) * 3.6)
        window = entry.get("data", {}).get("next_6_hours", {}).get("details", {})
        if "precipitation_amount" in window and moment.hour % 6 == 0:
            rainfall[day] += float(window["precipitation_amount"])

    ordered = sorted(day for day in temperatures if temperatures[day])
    if days:
        ordered = ordered[:days]
    frame = pd.DataFrame({
        "date": ordered,
        "max_c": [max(temperatures[d]) for d in ordered],
        "min_c": [min(temperatures[d]) for d in ordered],
        "rain_mm": [round(rainfall[d], 2) for d in ordered],
        "wind_kmh": [round(max(winds[d]), 1) if winds[d] else float("nan")
                     for d in ordered],
        "code": [None] * len(ordered),
    })
    frame["outlook"] = "not published by MET Norway"
    return frame[list(COLUMNS)]
