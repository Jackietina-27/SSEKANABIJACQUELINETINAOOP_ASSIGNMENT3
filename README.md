# Assignment 2 — Build C

Two mini-projects: a weather forecast client built on public web APIs, and a
Uganda tax engine. This is the third of three independent builds of the same
brief; it is not a variation on the other two but a different set of design
decisions, listed below.

```
python -m pytest -q                       # 162 tests, no network needed
python tools/assemble_notebooks.py        # rebuild both notebooks end to end
```

Everything runs offline. The weather tests replay sixteen recorded JSON
replies from `data/samples/`, and the notebooks fall back to those recordings
automatically when a live request fails, so **Restart & Run All** works with or
without a connection.

## Layout

```
src/common/request.py     RequestPlan, Reply, BackoffSchedule, execute()
src/common/errors.py      the failure hierarchy both projects share
src/weather/providers.py  ForecastProvider protocol, OpenMeteo, MetNorway, Recorder
src/weather/parsing.py    Location, locations_in(), *_to_daily(), COLUMNS
src/weather/codes.py      WMO 4677 code table
src/weather/display.py    summaries, tables, charts, house style
src/tax/money.py          Money, rounding_bias(), float_evidence()
src/tax/core.py           the abstract Tax contract, TaxContext, TaxResult, errors
src/tax/paye.py           Slice, Schedule, ScheduleSet, PayeTax
src/tax/ledger.py         Posting, Payslip, Payroll
src/tax/vat.py            Document, SupplyKind, VatTax
src/tax/business.py       CompanyTax, WithholdingTax, BusinessIncomeTax
src/tax/planner.py        the extension: employment against consultancy
tests/                    162 tests, including hypothesis property checks
notebooks/                the two deliverables, executed
tools/nb_src/             the notebook sources; notebooks are built from these
data/samples/             recorded API replies
figures/                  charts written by the notebooks
```

The notebooks are generated, not hand-edited. `tools/nb_src/*.py` are plain
Python files split into cells by `# ==[ markdown ]==` and `# ==[ code ]==`
banners; `tools/assemble_notebooks.py` converts them, executes every cell and
refuses to write a notebook that raised. That keeps the notebooks diffable and
guarantees the committed outputs are real.

## Mini-project 1: weather forecasts

**The central design choice: a request you can inspect before you send it.**
`RequestPlan` is a frozen value object holding a URL, query parameters and
headers. Building one performs no I/O — it can be printed, compared, used as a
cache key, or asserted against in a test. One function, `execute()`, turns a
plan into traffic. Everything else in the project is pure.

This is what makes the retry logic testable without a network. `BackoffSchedule`
is an iterable that yields delays and then stops; a test can list them. The
sleep function is a parameter, so a test can record the delays that *would*
have been taken. Nothing is mocked out.

**Providers are a `Protocol`, not a base class.** `OpenMeteo` and `MetNorway`
satisfy `ForecastProvider` structurally — they inherit nothing and share no
code, because they have none to share. `@runtime_checkable` lets the notebook
show the check passing. `Recorder`, which serves saved replies, satisfies the
same protocol without being a subclass of anything live: a subclass would
inherit a session and a timeout it has no use for, and would invite a call that
makes a real request from an object whose purpose is not to.

**Part A** times `urllib` and `requests` against the same endpoint, reporting
status, content type and elapsed time. The finding worth reading is in the
notebook: MET Norway's certificate chain fails under `urllib` on this machine
(`CERTIFICATE_VERIFY_FAILED`) and succeeds under `requests`, because `requests`
ships its own CA bundle while `urllib` uses the platform store. That is a real
reason to prefer one, not a matter of taste.

**Part B** geocodes a place name, fetches a daily forecast, decodes WMO codes
and produces a text summary, a table with units in the column names, and
charts.

**Part C** covers the failure paths: timeouts, `raise_for_status()`, capped
exponential backoff, and a hierarchy that separates temporary from permanent
failures so 4xx is never retried. Four failures are demonstrated against live
and recorded traffic: an unknown city, an out-of-range latitude (a genuine
400 saved in `forecast_invalid_latitude.json`), no network, and a timeout short
enough to always trip. MET Norway is the second source, and the notebook
measures how far the two models disagree instead of assuming they agree.

**Part D** is the written report: data flow with a diagram, licensing and
attribution for both APIs, reliability, and privacy — including why
`MetNorway` rounds coordinates to two decimal places before sending them.

### Contact details

MET Norway requires an identifying `User-Agent`. Nothing is hardcoded. The
client reads `WEATHER_CONTACT_EMAIL` from the environment and refuses a live
MET Norway request while it still holds the placeholder:

```
set WEATHER_CONTACT_EMAIL=your.name@students.mak.ac.ug      # Windows
export WEATHER_CONTACT_EMAIL=your.name@students.mak.ac.ug   # POSIX
```

Without it, the MET Norway cells replay recordings and say so. Open-Meteo needs
no key.

## Mini-project 2: Uganda tax engine

**`Money` truncates towards zero.** Builds of this kind usually round half-up.
This one does not, on the principle that the engine should never charge a
shilling the computation did not fully earn. The notebook then *measures* the
cost of that choice instead of asserting it is negligible: `rounding_bias()`
computes exact unrounded PAYE for 2,000 salaries and compares three rounding
policies. Truncation collects UGX 932 less than half-up across that sample —
under half a shilling per payslip, one-sided, and scaling with the payroll. A
rounding mode is a policy decision with a measurable revenue
consequence, and stating it without quantifying it seemed like the wrong
standard.

`Money` holds a `Decimal` and **refuses** floats at construction and in
multiplication instead of coercing them. `float_evidence()` returns three
verified failures used in the report: `round(2.675, 2)` giving `2.67`, the
non-associativity of float addition (which would make a payslip total depend on
the order its lines were added), and accumulated drift.

**Payroll is a ledger of postings, not a chain of subtractions.** Each
`Posting` records the base it was charged on. That makes the overlapping bases
visible in the data: LST comes off gross, PAYE is charged on gross *less* LST,
and employee NSSF is charged on the **full gross** — the LST deduction does not
reduce it. `Payroll.order()` prints the sequence with its justification, and
the notebook shows what getting the order wrong would cost.

**Schedules know when they stop.** Every `Schedule` carries an end date as well
as a start, so `ScheduleSet.in_force()` distinguishes three outcomes: covered,
before any schedule, and *in a gap between* schedules. A "most recent schedule
at or before this date" lookup cannot tell the last two apart and would
apply an expired schedule to a later year without complaint. `NoScheduleForDateError`
carries an `in_gap` flag so a caller can tell which happened.

**Other pieces.** VAT distinguishes zero-rated from exempt supplies and blocks
input recovery on the latter — the notebook quantifies the understatement that
conflating them produces. Corporate tax adds back disallowed costs and reports
what it added. `WithholdingTax` is modelled as a **credit** against a later
liability, not an extra charge.

### Where inheritance is used and where it is refused

`Tax` is an abstract base: `calculate(ctx: TaxContext) -> TaxResult`, no
arithmetic. Five classes implement it, and the notebook loops over all five.

Three refusals, each deliberate:

* **`BusinessIncomeTax` does not subclass `PayeTax`**, although a consultant is
  charged at exactly the same progressive rates. The base is fees less
  expenses and not gross pay, the assessment is annual, no NSSF or LST
  applies, and the withholding credit has no PAYE analogue — it would have
  inherited `calculate` only to override all of it. It composes a `PayeTax`
  and delegates the band arithmetic instead.
* **`Money` does not subclass `Decimal`.** Inheriting would give the arithmetic
  for free and also let `Money + 0.5` through an inherited operator. The point
  of the type is its ability to refuse, and a subclass cannot refuse what its
  parent permits.
* **`Recorder` does not subclass the live caller**, for the reason given above.

### Extension: employment or consultancy?

`src/tax/planner.py` compares putting someone on the payroll against engaging
them as a consultant, **holding the company's outlay constant** so one route
does not simply buy more labour.

The honest result required correcting my own first reading. The obvious
explanation — that the consultant's expense allowance drives the gap — is
wrong: `sensitivity()` shows consultancy keeping more cash at *every*
allowance, including zero. The actual driver is the 15% NSSF (5% employee plus
10% employer) that employment carries and consultancy does not. But NSSF is not
a tax; it is the worker's own money held for later. Counting it as value rather
than cost closes about two thirds of the cash gap, which is why `Comparison`
reports both measures and `caveats()` leads with the one that matters most:
whether a relationship *is* employment is a question of fact about control,
integration and risk, decided on substance and not on the label the parties
find convenient. URA can re-characterise an arrangement, with the tax falling
on the engager. The planner answers "what would each cost if both were
genuinely available", which is a narrower question than "which should we
choose".

## Testing

162 tests, no network, no display (matplotlib runs under `Agg`).

The brief asks for at least five assert-based checks on the parsing functions
using saved sample JSON; there are considerably more. The ones worth naming:

* Open-Meteo **omits the `results` key entirely** when nothing matches, rather
  than returning an empty list, and still answers 200. `locations_in()` uses
  `.get()` and raises `UnknownPlace`. `geocode_zero_results.json` is the real
  reply.
* A saved real 400 drives the invalid-latitude path, so the test exercises the
  same code a live failure would.
* `BackoffSchedule` delays are asserted as a list, and the retry cap is tested
  by counting calls with a recording sleep.
* PAYE is checked at all six band edges the brief names — 335,000; 335,001;
  410,000; 485,000; 10,000,000; 10,000,001 — and the marginal rate above ten
  million is asserted to be exactly 40%.
* `hypothesis` checks that tax is never negative, never exceeds income, and
  rises monotonically with income.

Two results that look wrong and are right: PAYE on 335,001 is zero, and PAYE on
10,000,001 equals PAYE on 10,000,000. Both are truncation of a sub-shilling
amount, which is why the threshold test asserts the *marginal rate* instead.

## Requirements

`requirements.txt`: requests, pandas, matplotlib, pytest, hypothesis, jupyter,
nbformat, nbclient.
