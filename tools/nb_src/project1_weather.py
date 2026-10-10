# ==[ markdown ]==
# # Mini-Project 1: Fetching and Displaying a Weather Forecast
#
# **The job.** Turn a place name into coordinates, fetch a forecast, show it
# clearly, and cross-check it against a second independent model.
#
# **The idea this build is organised around.** A request is **data before it
# is an action**. `RequestPlan` is a frozen description of what to call, with
# what parameters, under what timeout — and constructing one performs nothing.
# `execute()` is the only function that turns a plan into traffic. That falls
# out of the brief's own requirement to report a request's status, content type
# and timing: those belong to a *result*, and a result is easier to reason
# about when the thing that produced it could be inspected first.
#
# Providers sit behind a `typing.Protocol` instead of a base class, because
# Open-Meteo and MET Norway share no implementation at all — one reads a daily
# block straight out of the payload, the other rebuilds daily figures from an
# hourly series.
#
# **Running offline.** Every response used here was recorded into
# `data/samples/`. If the network is unavailable, or the contact email below
# is still the placeholder, the notebook replays them, so *Restart & Run All*
# always completes.

# ==[ markdown ]==
# ## Configuration
#
# The only cell to edit. The contact goes into the `User-Agent` that MET
# Norway requires, read from an environment variable first so a real address
# never has to enter the repository.

# ==[ code ]==
import os

# --- edit these ---------------------------------------------------------
PLACE = "Entebbe"
CONTACT_EMAIL = os.environ.get(
    "WEATHER_CONTACT_EMAIL", "<your.name@students.mak.ac.ug>")
# ------------------------------------------------------------------------

DAYS = 7
ATTEMPT_LIVE = True

HAVE_CONTACT = "@" in CONTACT_EMAIL and not CONTACT_EMAIL.startswith("<")
if HAVE_CONTACT:
    print(f"contact configured: {CONTACT_EMAIL}")
else:
    print("CONTACT_EMAIL is still the placeholder.")
    print("  MET Norway requires a genuine contact and blocks clients without")
    print("  one. Inventing an address would breach their terms of service, so")
    print("  this run replays the recorded responses instead. Set")
    print("  WEATHER_CONTACT_EMAIL, or edit the cell above, to go live.")

# ==[ code ]==
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import json
import time
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import requests

from src.common.errors import (AmbiguousName, RejectedRequest, Unreachable,
                               UnknownPlace, WeatherFailure)
from src.common.request import BackoffSchedule, RequestPlan, execute
from src.weather.codes import CODES, explain
from src.weather.display import (chart_comparison, chart_week, compare,
                                 house_style, readable_table, save, summarise)
from src.weather.parsing import (choose, locations_in, metno_to_daily,
                                 open_meteo_to_daily)
from src.weather.providers import (FORECAST, GEOCODE, METNO, MetNorway, OpenMeteo,
                                   PROVIDERS, Recorder, available_recordings,
                                   fetch, geocode, get_forecast, live_caller,
                                   user_agent)

house_style()
pd.set_option("display.width", 200)
print(f"{len(available_recordings())} recorded replies available")

# ==[ markdown ]==
# ### Choosing a caller
#
# A *caller* is anything that turns a plan into a reply. Live or recorded,
# everything downstream is written once.

# ==[ code ]==
REPLAY = Recorder()


def choose_caller():
    """A live caller if one works, otherwise the recorder."""
    if not (ATTEMPT_LIVE and HAVE_CONTACT):
        return REPLAY, "recorded (no contact configured)"
    candidate = live_caller(CONTACT_EMAIL,
                            BackoffSchedule(attempts=2, start=0.5))
    try:
        candidate(RequestPlan(GEOCODE, {"name": "Kampala", "count": 1}))
        return candidate, "live"
    except WeatherFailure as failure:
        print(f"live call failed ({type(failure).__name__}); replaying instead")
        return REPLAY, "recorded (network unavailable)"


CALLER, MODE = choose_caller()
print(f"mode  : {MODE}")

# ==[ markdown ]==
# ---
# # Part A: HTTP fundamentals
#
# ## A1. Geocoding with the standard library alone
#
# `urllib.parse.urlencode` builds the query string, `urllib.request.urlopen`
# performs the GET, `json` decodes it. The body arrives as **bytes**, and the
# charset must be read from the header and applied by hand — nothing does it
# for you, which is the first real difference between the two approaches.

# ==[ code ]==
PART_A_CITY = "Nairobi"


def geocode_by_hand(name: str, count: int = 5) -> dict:
    """Geocode using only urllib.request, urllib.parse and json."""
    query = urllib.parse.urlencode({"name": name, "count": count,
                                    "language": "en", "format": "json"})
    url = f"{GEOCODE}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=10) as reply:
        charset = reply.headers.get_content_charset() or "utf-8"
        text = reply.read().decode(charset)
        return {"url": url, "status": reply.status, "charset": charset,
                "content_type": reply.headers.get("Content-Type"),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "data": json.loads(text)}


if MODE == "live":
    outcome = geocode_by_hand(PART_A_CITY)
    print(f"url          : {outcome['url']}")
    print(f"status       : {outcome['status']}")
    print(f"content-type : {outcome['content_type']}")
    print(f"charset      : {outcome['charset']}  (applied by hand)")
    print(f"elapsed      : {outcome['elapsed_ms']:.0f} ms")
    leader = outcome["data"]["results"][0]
else:
    saved = REPLAY.load("geocode_nairobi")
    print("replaying the recorded reply:")
    print(f"url          : {saved.url}")
    print(f"status       : {saved.status}")
    print(f"content-type : {saved.content_type}")
    print(f"elapsed      : {saved.elapsed_ms:.0f} ms  (as recorded)")
    leader = saved.body["results"][0]

print(f"\nbest match   : {leader['name']}, {leader['country']} "
      f"({leader['latitude']}, {leader['longitude']})")

# ==[ markdown ]==
# ## A2. The same call with `requests`

# ==[ code ]==
if MODE == "live":
    started = time.perf_counter()
    reply = requests.get(GEOCODE,
                         params={"name": PART_A_CITY, "count": 5,
                                 "language": "en", "format": "json"},
                         headers={"User-Agent": user_agent(CONTACT_EMAIL)},
                         timeout=10)
    elapsed = (time.perf_counter() - started) * 1000
    reply.raise_for_status()
    data = reply.json()
    print(f"url          : {reply.url}")
    print(f"status       : {reply.status_code} {reply.reason}")
    print(f"content-type : {reply.headers['Content-Type']}")
    print(f"encoding     : {reply.encoding}  (taken from the header for you)")
    print(f"elapsed      : {elapsed:.0f} ms")
else:
    saved = REPLAY.load("geocode_nairobi")
    data = saved.body
    print("replaying; both approaches call the same endpoint")
    print(f"status       : {saved.status}")
    print(f"content-type : {saved.content_type}")
    print(f"elapsed      : {saved.elapsed_ms:.0f} ms  (as recorded)")

print(f"\nboth returned {len(data.get('results', []))} candidates")

# ==[ markdown ]==
# ## A3. Comparing the two approaches
#
# | | `urllib` | `requests` |
# |---|---|---|
# | **Query string** | `urlencode`, then concatenate | `params=` dict |
# | **A GET with headers and a timeout** | 8–10 lines | 2–3 lines |
# | **Decoding** | bytes; find the charset in the header and `.decode()` | `.text` / `.json()`, charset handled |
# | **Status** | raises `HTTPError` on 4xx/5xx, which is both an exception *and* a readable response, so it is easy to swallow | `.status_code` always present; `raise_for_status()` is explicit |
# | **Headers** | build a `Request` or an opener | `headers=` dict |
# | **Timeout** | one value for the whole operation | separable connect and read |
# | **Connection reuse** | none | `Session` pools them |
# | **Certificate trust** | the operating system's store | bundles `certifi` |
#
# **The row that mattered in practice.** The last one is not stylistic. While
# building this, the MET Norway endpoint failed through `urllib` with
# `SSL: CERTIFICATE_VERIFY_FAILED` and succeeded through `requests` on the
# same machine, same URL, same minute. `requests` ships the `certifi` CA
# bundle; `urllib` trusts the platform store, which on this Windows
# installation lacks the issuing authority. From the call site, "the service is
# down" and "my HTTP library cannot verify its certificate" produce the same
# symptom, and only trying a second client distinguishes them. Any
# availability monitoring built on a single HTTP client inherits that blind
# spot.
#
# **Where `urllib` still wins:** no third-party dependency, which matters in a
# minimal container or on a locked-down machine. For anything with retries,
# sessions or more than one endpoint, `requests` repays the dependency at once.

# ==[ markdown ]==
# ## A4. Status, content type and response time

# ==[ code ]==
probe_plan = OpenMeteo().plan(0.05611, 32.47972, 1)
print(f"the plan, before anything is sent: {probe_plan}")
print(f"cache key: {probe_plan.cache_key[:88]}...")
print()
probe = CALLER(probe_plan) if MODE == "live" else REPLAY.load("forecast_entebbe")
pd.Series(probe.details()).to_frame("value")

# ==[ markdown ]==
# ---
# # Part B: Forecast retrieval and display
#
# ## B1. `geocode(name)`, with both failure modes handled
#
# The first is a real trap. When nothing matches, the geocoding API returns
# **HTTP 200 with the `results` key absent** — not an empty list, not an error
# flag. `payload["results"]` raises `KeyError` on a successful response, and
# it does so wherever the payload is finally read rather than where the
# request was made.

# ==[ code ]==
print("a name matching nothing:")
try:
    geocode("Zzzqqxyznowhere", CALLER)
except UnknownPlace as problem:
    print(f"  UnknownPlace: {problem}")

empty = REPLAY.load("geocode_zero_results").body
print(f"\n  the entire payload: {empty}")
print(f"  is 'results' there? {'results' in empty}   <- the trap")

# ==[ code ]==
print("an ambiguous name:")
options = geocode("Springfield", CALLER, count=10)
for index, place in enumerate(options):
    print(f"  [{index}] {place}")

try:
    choose(options)
except AmbiguousName as problem:
    print(f"\n  AmbiguousName: {problem}")
    print("  -> silently taking the first is how 'Springfield' comes to mean")
    print("     Missouri to everyone who was not asked.")

print(f"\n  by index   : {choose(options, 0)}")
print(f"  by country : {choose(options, 'United States')}")

# ==[ code ]==
candidates = geocode(PLACE, CALLER)
for index, place in enumerate(candidates):
    print(f"  [{index}] {place}")
SPOT = choose(candidates) if len(candidates) == 1 else choose(candidates, 0)
print(f"\nusing: {SPOT}")
print(f"at two decimals (about a kilometre): {SPOT.rounded(2)}")

# ==[ markdown ]==
# ## B2–B3. `get_forecast(lat, lon, days)` into a DataFrame
#
# The request asks for daily maximum and minimum temperature, precipitation
# sum and maximum wind, with `timezone=auto` so the dates are the location's
# own calendar days and not UTC ones. The weather code is decoded through
# the WMO 4677 table.

# ==[ code ]==
forecast = get_forecast(SPOT.latitude, SPOT.longitude, DAYS, CALLER)
print(f"dates are local to {SPOT.described}")
forecast

# ==[ code ]==
pd.DataFrame([{"code": code, "description": text, "rain reaches the ground": wet}
              for code, (text, wet) in CODES.items()
              if code in (0, 3, 51, 61, 80, 95, 99)])

# ==[ markdown ]==
# ## B4. Summary, table and chart

# ==[ code ]==
print(summarise(forecast, SPOT.described, OpenMeteo().licence))

# ==[ code ]==
readable_table(forecast)

# ==[ code ]==
figure = chart_week(forecast, SPOT.described)
print("saved:", save(figure, "mp1_forecast.png").name)
plt.show()

# ==[ markdown ]==
# ---
# # Part C: Robustness and a second source
#
# ## C1. One guarded function
#
# `execute()` sets a timeout, calls `raise_for_status()`, retries transient
# failures on an exponential schedule with a ceiling, and raises a named
# exception when it finally gives up. The schedule is an ordinary iterable, so
# it can be listed and summed without waiting a single second.

# ==[ code ]==
schedule = BackoffSchedule(attempts=4, start=0.5, multiplier=2.0, cap=4.0)
print(f"schedule   : {schedule}")
print(f"waits      : {schedule.waits()} seconds")
print(f"worst case : {schedule.worst_case()} s of waiting")
print()
print("what is retried, and what is not:")
print("  408 / 425 / 429 / 5xx / timeout -> retry; the condition may clear")
print("  any other 4xx                   -> raise now; the request is wrong")
print()
print("Retrying an out-of-range latitude four times costs four round trips and")
print("delays the message the caller actually needs.")

# ==[ markdown ]==
# ## C2. Four deliberately broken inputs

# ==[ code ]==
print("1. an invalid city")
try:
    geocode("Zzzqqxyznowhere", CALLER)
except UnknownPlace as problem:
    print(f"   {type(problem).__name__}: {problem}")

# ==[ code ]==
print("2. an invalid latitude")
print("   (a) rejected before a plan is even built:")
try:
    get_forecast(999, 32.0, 7, CALLER)
except ValueError as problem:
    print(f"       ValueError: {problem}")

print("   (b) and as the API rejects it, from a recorded 400:")
try:
    REPLAY.load("forecast_invalid_latitude")
except RejectedRequest as problem:
    print(f"       {type(problem).__name__}: {problem}")
    print(f"       status {problem.status}; not retried")

# ==[ code ]==
print("3. no network (an unroutable address stands in)")
started = time.perf_counter()
try:
    execute(RequestPlan("https://10.255.255.1/forecast", timeout=2.0),
            BackoffSchedule(attempts=3, start=0.05))
except Unreachable as problem:
    print(f"   {type(problem).__name__}: {problem}")
    print(f"   {problem.attempts} attempts in "
          f"{(time.perf_counter() - started):.1f} s")

# ==[ code ]==
print("4. a timeout too small for any reply")
try:
    execute(RequestPlan(GEOCODE, {"name": "Kampala", "count": 1}, timeout=0.001),
            BackoffSchedule(attempts=2, start=0.05))
    print("   unexpectedly succeeded; the network was very fast")
except Unreachable as problem:
    print(f"   {type(problem).__name__}: {problem}")
    print(f"   underlying cause: {type(problem.cause).__name__}")

# ==[ markdown ]==
# ## C3. The second source
#
# Open-Meteo publishes daily aggregates. MET Norway publishes an **hourly**
# series, so the daily figures are rebuilt here: maximum and minimum of the
# hourly air temperature, maximum hourly wind, and the six-hourly
# precipitation blocks sampled every sixth hour so the overlapping windows are
# not counted twice. MET Norway reports wind in **m/s**, converted to km/h —
# without which the second source would appear to contradict the first by a
# factor of 3.6.
#
# Note also that `MetNorway.plan` rounds the coordinates to two decimals. That
# is a deliberate privacy choice, visible in the plan before anything is sent,
# and it is discussed in Part D.

# ==[ code ]==
print("the plan MET Norway will receive:")
print(" ", MetNorway().plan(SPOT.latitude, SPOT.longitude, DAYS).params)
print(f"  (the precise position is {SPOT.latitude:.6f}, {SPOT.longitude:.6f})")

try:
    second, reply = fetch(PROVIDERS["MET Norway"], CALLER,
                          SPOT.latitude, SPOT.longitude, DAYS)
    print(f"\nMET Norway: {reply!r}")
except Exception as problem:
    print(f"\nlive MET Norway unavailable ({type(problem).__name__}); replaying")
    second, reply = fetch(PROVIDERS["MET Norway"], REPLAY,
                          SPOT.latitude, SPOT.longitude, DAYS)

second

# ==[ code ]==
agreement = compare(forecast, second)
print(repr(agreement))
agreement.overview().round(2)

# ==[ code ]==
for measure in ("max_c", "rain_mm"):
    print(" ", agreement.reads_as(measure, tolerance=2.0))
print()
print("Temperature agrees closely; rainfall does not, and that is the")
print("expected result rather than a defect. Temperature is a smooth field")
print("both models resolve well. Tropical rainfall is convective and")
print("sub-grid — it depends on where an individual storm cell forms, which")
print("neither model can place. The two daily totals are also assembled")
print("differently: Open-Meteo sums its own daily field, while the MET Norway")
print("figure is rebuilt here from six-hourly blocks.")

# ==[ code ]==
figure = chart_comparison(agreement)
print("saved:", save(figure, "mp1_two_models.png").name)
plt.show()

# ==[ markdown ]==
# ---
# # Part D: Report
#
# ## The data flow
#
# ```
#   PLACE                            CONTACT_EMAIL (env var, never committed)
#     |                                        |
#     v                                        v
#   +-----------------------------------------------------------------+
#   |  geocode(name)  /  get_forecast(lat, lon, days)                  |
#   +-----------------------------------------------------------------+
#     |
#     |  builds, but does not send:
#     v
#   +--------------------+        inspectable, comparable, loggable,
#   |    RequestPlan     |------> testable -- nothing has happened yet
#   |  url + params +    |
#   |  headers + timeout |
#   +--------------------+
#     |
#     v
#   +=================================================================+
#   |  execute(plan, BackoffSchedule)        OR       Recorder(plan)  |
#   |   timeout, raise_for_status, capped exponential backoff         |
#   |   4xx -> RejectedRequest (never retried)                        |
#   |   exhausted -> Unreachable                                      |
#   +=================================================================+
#     |
#     v
#   +----------+
#   |  Reply   |  status, content-type, elapsed_ms, attempts, body
#   +----------+
#     |
#     v
#   +-----------------------------------------------------------------+
#   |  locations_in()  -- .get("results"), never ["results"]           |
#   |     nothing matched -> UnknownPlace                              |
#   |     several matched -> AmbiguousName -> choose()                 |
#   +-----------------------------------------------------------------+
#     |
#     v
#   +------------+
#   |  Location  |
#   +------------+
#     |
#     +---------------------------+
#     v                           v
#   OpenMeteo.plan/read     MetNorway.plan/read
#   daily block             hourly -> rolled up; m/s -> km/h
#     |                           |  (coords rounded to 2dp here)
#     +------------+--------------+
#                  v
#           compare() -> Comparison
#                  |
#     +------------+------------+
#     v            v            v
#  summarise  readable_table  chart_week / chart_comparison
# ```
#
# Two features of that shape are worth naming. The **plan is separate from the
# call**, so the parameters a provider will receive — including the rounded
# coordinates — can be examined before anything leaves the machine. And
# **parsing is separate from fetching**: `locations_in` and
# `open_meteo_to_daily` take a dictionary and return a table, touching no
# network and no clock. That is why all 69 weather tests run unplugged, and it
# is where the genuine defect lived: the missing-`results` problem is a parsing
# bug, and no amount of retry logic would ever have caught it.
#
# ## Licensing, and what it means for a real product
#
# **Open-Meteo** publishes under CC BY 4.0, free for non-commercial use with no
# key, with commercial use on a paid plan. The attribution obligation is why
# `licence` is a field on the provider rather than a comment: if the credit
# lives somewhere the code does not carry, a refactor ships a product that
# breaches its licence silently. The free tier is bounded by fair use rather
# than a published quota, which is fine for coursework and unusable as a
# planning assumption — "we will throttle you if you are unreasonable" is not a
# number you can size a cache against.
#
# **MET Norway** publishes under CC BY 4.0 and the Norwegian Licence for Open
# Government Data, with three conditions that bite harder. A descriptive
# `User-Agent` carrying genuine contact details is mandatory and enforced by
# blocking. Clients are asked to honour the `Expires` header and use
# `If-Modified-Since` instead of re-polling. And throughput is capped around
# twenty requests a second. The contact requirement is exactly why this
# notebook declines to make live MET Norway calls until a real address is set:
# supplying a fabricated one would breach the terms, and the brief prohibits it
# independently.
#
# For a product, three consequences follow. Attribution belongs in the
# interface, not in a licence file nobody opens. Caching stops being an
# optimisation and becomes a condition of use — the `Expires` header states
# precisely how long a reply remains valid, and ignoring it while scaling is
# the ordinary route to being blocked. And relying on a free tier is a
# commercial exposure rather than a technical one: terms change, and a product
# with one provider has a single point of failure that no retry policy
# addresses. The two-source design here is half correctness check and half
# insurance.
#
# ## Reliability, and how the code answers it
#
# **A 200 that means failure.** The missing `results` key is the clearest
# demonstration that HTTP status and application success are different
# questions. The fix is one character — `.get` and not `[]` — and a named
# exception raised at the point the problem occurs.
#
# **Two libraries disagreeing about reachability.** The TLS divergence meant a
# live endpoint looked dead. A retry policy would have made this *worse*: three
# further attempts, three identical certificate failures, and a confident
# report of an outage that was not happening. Retries cannot distinguish a
# problem at the far end from a problem in the client.
#
# **Transient against permanent.** `execute()` separates them explicitly. A 503
# is repeated; a 400 raises `RejectedRequest` at once, carrying the provider's
# own `reason` string so the caller sees *why* rather than only *that*.
#
# **Unbounded waiting looks like a hang.** Exponential growth without a ceiling
# can leave a notebook apparently frozen with no way to tell whether waiting
# will help. `BackoffSchedule` caps each wait and the attempt count, and
# `worst_case()` states the bound before anything runs.
#
# ## Privacy
#
# A location query is not neutral telemetry. Three things travel together: the
# place name or coordinates in the query string, the client's IP address, and —
# because MET Norway requires it — an email address in the `User-Agent`. That
# combination links an identifiable person to an interest in a specific place
# at a specific time, held by a third party in another jurisdiction.
#
# The provider's reason is sound: MET Norway wants to reach the operator of a
# misbehaving client. But the effect is that an identifier accompanies every
# request, including those where nobody asked for it. Open-Meteo requires no
# such header, and a careless client sends the same `User-Agent` to both — so
# the cheapest available privacy improvement is simply not transmitting data a
# provider did not request.
#
# The sharper issue is **resolution**, and this build does something about it
# instead of only describing it. A forecast request carries coordinates, not a
# name. Four decimal places locates a person to about ten metres; two places to
# about a kilometre. No weather model resolves anything near ten metres, so the
# extra precision buys the user nothing and costs them a far more exact
# position. `MetNorway.plan` therefore rounds to two decimals, and because the
# plan is a value that can be printed before it is sent, the cell in Part C
# shows exactly what will leave the machine. The trade-off is visible rather
# than buried, which is the most useful thing a small design decision can be.
#
# Aggregation is where the real exposure lies. One query for "Entebbe"
# discloses almost nothing; a user's query history over months discloses where
# they live, when they travel and where they are going next. Caching helps
# twice over here — it reduces the number of requests and coarsens the timing
# signal a provider can observe — and it is already required by MET Norway's
# terms, so the privacy interest and the licensing obligation point the same
# way.
#
# One point specific to this submission: the recordings in `data/samples/`
# contain only public forecast values and no personal data, which is why they
# are safe to commit. A project that cached *real user queries* to make its
# tests reproducible would have manufactured a privacy problem in the act of
# solving an engineering one.

# ==[ markdown ]==
# ## Offline checks
#
# The brief asks for at least five assert-based checks on the parsing
# functions, against saved sample JSON. The full suite is 69 tests in
# `tests/test_weather.py`; a representative set runs inline here.

# ==[ code ]==
saved = {stem: REPLAY.load(stem).body
         for stem in ("geocode_entebbe", "geocode_zero_results",
                      "forecast_entebbe", "metno_entebbe")}

# 1. a good geocoding payload yields usable coordinates
places = locations_in(saved["geocode_entebbe"], "Entebbe")
assert places and places[0].country == "Uganda"
assert -90 <= places[0].latitude <= 90 and -180 <= places[0].longitude <= 180

# 2. the absent-results case raises rather than throwing KeyError
assert "results" not in saved["geocode_zero_results"]
try:
    locations_in(saved["geocode_zero_results"], "nowhere")
    raise AssertionError("should have raised UnknownPlace")
except UnknownPlace:
    pass

# 3. the forecast table is the right shape and internally consistent
table = open_meteo_to_daily(saved["forecast_entebbe"])
assert len(table) == 7
assert (table["max_c"] >= table["min_c"]).all()
assert (table["rain_mm"] >= 0).all()

# 4. codes decode, and an unpublished one is not quietly called clear
assert explain(95) == "Thunderstorm"
assert explain(7) != "Clear sky" and "code 7" in explain(7)

# 5. MET Norway rolls up to the same columns, converted to km/h
rolled = metno_to_daily(saved["metno_entebbe"], days=7)
assert list(rolled.columns) == list(table.columns)
assert (rolled["max_c"] >= rolled["min_c"]).all()

# 6. a plan is inert: building one sends nothing
plan = OpenMeteo().plan(0.05611, 32.47972, 7)
assert plan.params["forecast_days"] == 7 and plan.timeout > 0

# 7. and the two sources broadly agree on temperature
assert compare(table, rolled).gap("max_c").abs().max() < 5.0

print("seven inline checks passed, using only recorded sample JSON")
print(f"the full suite puts {len(available_recordings())} recordings through "
      f"69 tests with no network")
