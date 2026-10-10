"""Tests for Mini-Project 1. All offline, against the recorded replies."""
import json

import pandas as pd
import pytest
import requests

from src.common.errors import (AmbiguousName, MalformedPayload, RejectedRequest,
                               TemporaryFailure, Unreachable, UnknownPlace,
                               WeatherFailure)
from src.common.request import BackoffSchedule, Reply, RequestPlan, execute
from src.weather.codes import CODES, explain, known, precipitating
from src.weather.display import chart_week, compare, readable_table, summarise
from src.weather.parsing import (COLUMNS, Location, choose, locations_in,
                                 metno_to_daily, open_meteo_to_daily)
from src.weather.providers import (FORECAST, GEOCODE, ForecastProvider, MetNorway,
                                   OpenMeteo, PROVIDERS, Recorder,
                                   available_recordings, fetch, geocode,
                                   get_forecast, live_caller, user_agent)


# ======================================================================
# Request plans -- data before action
# ======================================================================
def test_a_plan_is_inert_until_executed():
    """Constructing a plan performs nothing; that is the point."""
    plan = RequestPlan(GEOCODE, {"name": "Entebbe", "count": 5}, label="geocoding")
    assert plan.url == GEOCODE
    assert "Entebbe" in plan.cache_key
    assert "geocoding" in str(plan)


def test_a_plan_is_frozen_and_copies_cleanly():
    plan = RequestPlan(GEOCODE, {"name": "x"})
    with_header = plan.with_headers(**{"User-Agent": "test"})
    assert with_header.headers["User-Agent"] == "test"
    assert plan.headers == {}                  # the original is untouched
    with pytest.raises(Exception):
        plan.url = "https://other"


@pytest.mark.parametrize("kwargs", [{"url": "not-a-url"}, {"url": ""},
                                    {"timeout": 0}, {"timeout": -1}])
def test_impossible_plans_rejected(kwargs):
    base = {"url": "https://example.invalid", "timeout": 5.0}
    with pytest.raises(ValueError):
        RequestPlan(**{**base, **kwargs})


def test_the_cache_key_is_order_independent():
    first = RequestPlan(GEOCODE, {"a": 1, "b": 2})
    second = RequestPlan(GEOCODE, {"b": 2, "a": 1})
    assert first.cache_key == second.cache_key


# ======================================================================
# Backoff
# ======================================================================
def test_the_schedule_is_exponential_and_capped():
    schedule = BackoffSchedule(attempts=5, start=1.0, multiplier=2.0, cap=4.0)
    assert schedule.waits() == [1.0, 2.0, 4.0, 4.0]
    assert schedule.worst_case() == 11.0
    assert len(schedule) == 4


def test_a_single_attempt_never_waits():
    assert BackoffSchedule(attempts=1).waits() == []


def test_the_schedule_is_iterable():
    assert list(BackoffSchedule(attempts=3, start=0.5)) == [0.5, 1.0]


@pytest.mark.parametrize("kwargs", [
    {"attempts": 0}, {"start": -1}, {"multiplier": 0.5},
    {"start": 10, "cap": 1}])
def test_impossible_schedules_rejected(kwargs):
    with pytest.raises(ValueError):
        BackoffSchedule(**kwargs)


class _Session:
    """A session that fails a set number of times, then succeeds."""

    def __init__(self, failures: int, status: int = 503):
        self.failures, self.status, self.calls = failures, status, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        reply = requests.Response()
        reply.url = url
        reply.headers["Content-Type"] = "application/json"
        if self.calls <= self.failures:
            reply.status_code, reply.reason = self.status, "Unavailable"
            reply._content = b"{}"
        else:
            reply.status_code, reply.reason = 200, "OK"
            reply._content = b'{"ok": true}'
        return reply


class _Rejecting:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        reply = requests.Response()
        reply.url = url
        reply.status_code, reply.reason = 400, "Bad Request"
        reply.headers["Content-Type"] = "application/json"
        reply._content = json.dumps(
            {"error": True, "reason": "Latitude must be in range"}).encode()
        return reply


def test_execute_retries_a_transient_failure():
    session = _Session(failures=2)
    waits = []
    reply = execute(RequestPlan("https://example.invalid/x"),
                    BackoffSchedule(attempts=3, start=0.01),
                    session=session, sleep=waits.append)
    assert reply.body == {"ok": True}
    assert reply.attempts == 3 and session.calls == 3 and len(waits) == 2


def test_execute_gives_up_with_a_named_error():
    with pytest.raises(Unreachable) as caught:
        execute(RequestPlan("https://example.invalid/x"),
                BackoffSchedule(attempts=3, start=0.01),
                session=_Session(failures=99), sleep=lambda _: None)
    assert caught.value.attempts == 3
    assert isinstance(caught.value, WeatherFailure)


def test_a_rejected_request_is_never_retried():
    session = _Rejecting()
    with pytest.raises(RejectedRequest) as caught:
        execute(RequestPlan("https://example.invalid/x"),
                BackoffSchedule(attempts=4, start=0.01),
                session=session, sleep=lambda _: None)
    assert session.calls == 1
    assert caught.value.status == 400
    assert "Latitude" in (caught.value.reason or "")


def test_user_agent_requires_a_contact():
    assert "me@uni.ac.ug" in user_agent("me@uni.ac.ug")
    with pytest.raises(ValueError):
        user_agent("")


# ======================================================================
# Parsing -- the offline checks the brief asks for
# ======================================================================
def test_a_recorded_geocode_parses(body):
    places = locations_in(body("geocode_entebbe"), "Entebbe")
    assert places and places[0].country == "Uganda"
    assert -90 <= places[0].latitude <= 90


def test_the_absent_results_key_raises_not_key_error(body):
    """No matches omits 'results' from an otherwise ordinary 200."""
    payload = body("geocode_zero_results")
    assert "results" not in payload
    with pytest.raises(UnknownPlace):
        locations_in(payload, "Zzzqqxyznowhere")


def test_a_malformed_candidate_does_not_lose_the_others():
    payload = {"results": [
        {"name": "Good", "latitude": 1.0, "longitude": 2.0, "country": "X"},
        {"name": "Missing coordinates"},
        {"name": "Also good", "latitude": 3.0, "longitude": 4.0}]}
    assert [p.name for p in locations_in(payload, "x")] == ["Good", "Also good"]


def test_the_forecast_table_is_sane(body):
    table = open_meteo_to_daily(body("forecast_entebbe"))
    assert list(table.columns) == list(COLUMNS)
    assert len(table) == 7
    assert (table["max_c"] >= table["min_c"]).all()
    assert (table["rain_mm"] >= 0).all()
    assert (table["wind_kmh"] >= 0).all()


def test_weather_codes_are_decoded(body):
    table = open_meteo_to_daily(body("forecast_entebbe"))
    assert table["outlook"].notna().all()
    assert not table["outlook"].str.startswith("Unknown").any()


@pytest.mark.parametrize("payload", [
    {"latitude": 1.0}, {"daily": {"time": ["2026-01-01"]}},
    {"daily": {"time": ["2026-01-01"], "temperature_2m_max": [20.0]}}])
def test_bad_forecast_payloads_rejected(payload):
    with pytest.raises(MalformedPayload):
        open_meteo_to_daily(payload)


def test_metno_rolls_up_to_the_same_shape(body):
    table = metno_to_daily(body("metno_entebbe"), days=7)
    assert list(table.columns) == list(COLUMNS)
    assert len(table) == 7
    assert (table["max_c"] >= table["min_c"]).all()


def test_metno_wind_converts_from_metres_per_second(body):
    payload = body("metno_entebbe")
    raw = [e["data"]["instant"]["details"]["wind_speed"]
           for e in payload["properties"]["timeseries"]
           if "wind_speed" in e["data"]["instant"]["details"]]
    table = metno_to_daily(payload, days=7)
    assert max(raw) < 30                       # plausible as m/s
    assert table["wind_kmh"].max() == pytest.approx(max(raw) * 3.6, rel=0.02)


def test_metno_without_a_timeseries_is_rejected():
    with pytest.raises(MalformedPayload):
        metno_to_daily({"properties": {}})


def test_the_two_sources_agree_on_temperature(body):
    first = open_meteo_to_daily(body("forecast_entebbe"))
    second = metno_to_daily(body("metno_entebbe"), days=7)
    assert compare(first, second).gap("max_c").abs().max() < 5.0


# ======================================================================
# Locations
# ======================================================================
@pytest.mark.parametrize("lat,lon", [(999, 0), (0, 400), (-91, 0)])
def test_impossible_coordinates_rejected(lat, lon):
    with pytest.raises(ValueError):
        Location("Nowhere", "X", lat, lon)


def test_rounding_reduces_what_is_transmitted():
    """Two decimals is about a kilometre; four is about ten metres."""
    place = Location("Entebbe", "Uganda", 0.056112, 32.479721)
    assert place.rounded(2) == (0.06, 32.48)
    assert place.rounded(4) != place.rounded(2)


def test_described_reads_well():
    place = Location("Entebbe", "Uganda", 0.0561, 32.4797, area="Central")
    assert place.described == "Entebbe, Central, Uganda"


def test_ambiguity_raises_rather_than_guessing(replay):
    options = geocode("Springfield", replay, count=10)
    assert len(options) > 1
    with pytest.raises(AmbiguousName) as caught:
        choose(options)
    assert len(caught.value.candidates) == len(options)


def test_ambiguity_resolves_by_index_or_country(replay):
    options = geocode("Springfield", replay, count=10)
    assert choose(options, 0) is options[0]
    assert choose(options, "United States").country == "United States"
    with pytest.raises(IndexError):
        choose(options, 99)
    with pytest.raises(UnknownPlace):
        choose(options, "Atlantis")


# ======================================================================
# Weather codes
# ======================================================================
@pytest.mark.parametrize("code,text", [
    (0, "Clear sky"), (3, "Overcast"), (61, "Slight rain"),
    (95, "Thunderstorm"), (99, "Thunderstorm with heavy hail")])
def test_published_codes_decode(code, text):
    assert explain(code) == text


def test_unknown_codes_are_not_silently_called_clear():
    assert explain(7) != "Clear sky"
    assert "code 7" in explain(7)
    assert explain(None) == "Not reported"
    assert explain("rubbish") == "Not reported"
    assert known(7) is False and known(95) is True


@pytest.mark.parametrize("code,wet", [(0, False), (3, False), (51, True),
                                      (61, True), (80, True), (95, True)])
def test_precipitating(code, wet):
    assert precipitating(code) is wet


def test_the_table_has_no_duplicate_descriptions():
    descriptions = [text for text, _ in CODES.values()]
    assert len(set(descriptions)) == len(descriptions)


# ======================================================================
# Providers
# ======================================================================
def test_both_providers_satisfy_the_protocol():
    for provider in PROVIDERS.values():
        assert isinstance(provider, ForecastProvider)
        assert provider.name and provider.licence


def test_providers_produce_the_same_table_shape(replay):
    for provider in PROVIDERS.values():
        table, reply = fetch(provider, replay, 0.05611, 32.47972, 5)
        assert list(table.columns) == list(COLUMNS)
        assert isinstance(reply, Reply) and reply.recorded


def test_metno_plans_with_reduced_precision():
    """The privacy choice is visible in the plan, before anything is sent."""
    plan = MetNorway().plan(0.056112, 32.479721, 7)
    assert plan.params["lat"] == 0.06
    assert plan.params["lon"] == 32.48


def test_open_meteo_rejects_an_impossible_horizon():
    with pytest.raises(ValueError):
        OpenMeteo().plan(0.05, 32.47, 0)
    with pytest.raises(ValueError):
        OpenMeteo().plan(0.05, 32.47, 40)


def test_geocode_needs_a_name(replay):
    with pytest.raises(ValueError):
        geocode("", replay)


@pytest.mark.parametrize("lat,lon", [(999, 32.0), (0.3, 400.0)])
def test_get_forecast_validates_coordinates(replay, lat, lon):
    with pytest.raises(ValueError):
        get_forecast(lat, lon, 7, replay)


def test_get_forecast_honours_the_day_count(replay):
    assert len(get_forecast(0.05611, 32.47972, 3, replay)) == 3
    assert len(get_forecast(0.05611, 32.47972, 7, replay)) == 7


def test_a_recorded_400_replays_as_a_rejection(replay):
    with pytest.raises(RejectedRequest):
        replay.load("forecast_invalid_latitude")


def test_an_unknown_plan_is_reported_not_invented(replay):
    with pytest.raises(Unreachable):
        replay(RequestPlan(GEOCODE, {"name": "Atlantis"}))


def test_every_recording_is_present():
    assert len(available_recordings()) == 16


def test_the_recorder_is_not_live(replay):
    assert replay.live is False
    assert len(replay) == 16


# ======================================================================
# Comparison and display
# ======================================================================
def test_comparison_uses_only_shared_dates(replay):
    first, _ = fetch(PROVIDERS["Open-Meteo"], replay, 0.05611, 32.47972, 7)
    second, _ = fetch(PROVIDERS["MET Norway"], replay, 0.05611, 32.47972, 5)
    result = compare(first, second)
    assert len(result) <= 5
    assert "max_c" in result.measures()


def test_comparison_reports_a_verdict(replay):
    first, _ = fetch(PROVIDERS["Open-Meteo"], replay, 0.05611, 32.47972, 7)
    second, _ = fetch(PROVIDERS["MET Norway"], replay, 0.05611, 32.47972, 7)
    result = compare(first, second)
    assert "max_c" in result.reads_as("max_c")
    assert not result.overview().empty


def test_an_unknown_measure_raises(replay):
    first, _ = fetch(PROVIDERS["Open-Meteo"], replay, 0.05611, 32.47972, 3)
    second, _ = fetch(PROVIDERS["MET Norway"], replay, 0.05611, 32.47972, 3)
    with pytest.raises(KeyError):
        compare(first, second).gap("humidity")


def test_summary_names_the_place(replay):
    table = get_forecast(0.05611, 32.47972, 7, replay)
    text = summarise(table, "Entebbe, Uganda", OpenMeteo().licence)
    assert "Entebbe" in text and "°C" in text and "mm" in text


def test_the_display_table_carries_units(replay):
    shown = readable_table(get_forecast(0.05611, 32.47972, 7, replay))
    assert "Max (°C)" in shown.columns and "Rain (mm)" in shown.columns
    assert "code" not in shown.columns


def test_the_chart_is_labelled(replay):
    figure = chart_week(get_forecast(0.05611, 32.47972, 7, replay), "Entebbe")
    axis = figure.axes[0]
    assert axis.get_xlabel() and axis.get_ylabel()
    assert "Entebbe" in axis.get_title()


def test_charting_nothing_raises():
    with pytest.raises(ValueError):
        chart_week(pd.DataFrame(columns=list(COLUMNS)), "Nowhere")
