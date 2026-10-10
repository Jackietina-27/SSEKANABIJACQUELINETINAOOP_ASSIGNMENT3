"""PAYE: progressive slices, and schedules that know when they stop.

The charge is built by iterating over slices of income and summing what each
contributes. That is slower than a precomputed cumulative table, and it is
what the statute actually says, so a reader can hold the code and the Third
Schedule side by side.

Schedules carry both a start and an end date, which is what lets ScheduleSet
tell three failures apart: a date before any schedule, a date after the last
closed one, and a date in a gap between two. A "most recent schedule at or
before this date" lookup cannot, and would apply an expired schedule to a
later year while looking perfectly confident about it.

The resident schedule is the brief's. The non-resident one is not in the
brief; it comes from URA's monthly tables, where a non-resident gets no nil
band and pays 10% from the first shilling. Both carry the additional 10%
above UGX 10,000,000, which makes the top marginal rate 40%.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator, Sequence

import pandas as pd

from src.tax.core import (BadRate, Entry, InvalidIncomeError,
                          NoScheduleForDateError, Residency, Tax, TaxContext,
                          TaxResult)
from src.tax.money import Money

__all__ = ["EXTRA_RATE", "EXTRA_THRESHOLD", "PayeTax", "Schedule", "ScheduleSet",
           "Slice", "SCHEDULES"]

# The additional charge above UGX 10,000,000 a month.
EXTRA_THRESHOLD = Money("10000000")
EXTRA_RATE = Decimal("0.10")


@dataclass(frozen=True)
class Slice:
    """A band of income and the rate that applies to it."""

    starts_above: Money
    ends_at: Money | None
    rate: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.rate, Decimal):
            raise BadRate(self.rate, "a slice rate must be a Decimal")
        if not Decimal("0") <= self.rate <= Decimal("1"):
            raise BadRate(self.rate)
        if self.ends_at is not None and self.ends_at <= self.starts_above:
            raise ValueError(
                f"a slice must end above where it starts: "
                f"{self.ends_at} <= {self.starts_above}")

    def portion_of(self, income: Money) -> Money:
        """How much of income falls inside this slice."""
        if income <= self.starts_above:
            return Money.none()
        top = income if self.ends_at is None else min(income, self.ends_at)
        return top - self.starts_above

    def charge_on(self, income: Money) -> Money:
        """What this slice contributes to the total."""
        return self.portion_of(income) * self.rate

    def __str__(self) -> str:
        ceiling = "and above" if self.ends_at is None else f"to {self.ends_at}"
        return f"{self.starts_above} {ceiling} at {self.rate:.0%}"


@dataclass(frozen=True)
class Schedule:
    """A full set of slices, valid between two dates."""

    residency: str
    opens: dt.date
    closes: dt.date | None
    slices: tuple[Slice, ...]
    authority: str = ""
    extra_above: Money = EXTRA_THRESHOLD
    extra_rate: Decimal = EXTRA_RATE

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("a schedule needs at least one slice")
        if self.slices[0].starts_above != Money.none():
            raise ValueError("the first slice must start at zero")
        for earlier, later in zip(self.slices, self.slices[1:]):
            if earlier.ends_at is None:
                raise ValueError("only the last slice may be open-ended")
            if earlier.ends_at != later.starts_above:
                raise ValueError(
                    f"slices must be contiguous: {earlier.ends_at} != "
                    f"{later.starts_above}")
        if self.slices[-1].ends_at is not None:
            raise ValueError("the last slice must be open-ended")
        if self.closes is not None and self.closes < self.opens:
            raise ValueError("a schedule cannot close before it opens")

    def covers(self, when: dt.date) -> bool:
        return self.opens <= when and (self.closes is None or when <= self.closes)

    def __iter__(self) -> Iterator[Slice]:
        return iter(self.slices)

    def __len__(self) -> int:
        return len(self.slices)

    @property
    def span(self) -> str:
        return (f"{self.residency} {self.opens.isoformat()} to "
                f"{self.closes.isoformat() if self.closes else 'current'}")

    def __str__(self) -> str:
        return self.span

    def charge(self, income: Money) -> tuple[Money, tuple[Entry, ...]]:
        """Total tax and the slice-by-slice working.

        Each slice is applied to its own portion and the results summed, which
        is what the statute describes. The nil band appears in the working even
        though it contributes nothing, so a reader can see how much income fell
        in it.
        """
        if income.is_negative():
            raise InvalidIncomeError(income)
        total, working = Money.none(), []
        for band in self.slices:
            portion = band.portion_of(income)
            if portion.is_none():
                continue
            contribution = portion * band.rate
            total = total + contribution
            working.append(Entry(f"slice {band}", portion, band.rate,
                                 contribution))

        if income > self.extra_above:
            over = income - self.extra_above
            additional = over * self.extra_rate
            total = total + additional
            working.append(Entry(f"additional charge above {self.extra_above}",
                                 over, self.extra_rate, additional))
        return total, tuple(working)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "above": int(band.starts_above.amount),
            "up to": "—" if band.ends_at is None else int(band.ends_at.amount),
            "rate": f"{band.rate:.0%}"} for band in self.slices])


def _slice(low: str, high: str | None, rate: str) -> Slice:
    return Slice(Money(low), None if high is None else Money(high), Decimal(rate))


RESIDENT_AUTHORITY = (
    "URA monthly PAYE for resident individuals. Reproduces the baseline table "
    "in the assignment brief; Income Tax Act Cap 340, Third Schedule, as "
    "published at ura.go.ug.")

NON_RESIDENT_AUTHORITY = (
    "URA monthly PAYE for non-resident individuals: no nil band, 10% from the "
    "first shilling. Not given in the brief; taken from URA's published "
    "monthly tables, Income Tax Act Cap 340, Third Schedule Part II.")

# Every schedule this engine knows about.
SCHEDULES: tuple[Schedule, ...] = (
    Schedule(Residency.RESIDENT, dt.date(2012, 7, 1), None,
             (_slice("0", "335000", "0.00"), _slice("335000", "410000", "0.20"),
              _slice("410000", "485000", "0.25"), _slice("485000", None, "0.30")),
             RESIDENT_AUTHORITY),
    Schedule(Residency.NON_RESIDENT, dt.date(2012, 7, 1), None,
             (_slice("0", "335000", "0.10"), _slice("335000", "410000", "0.20"),
              _slice("410000", "485000", "0.25"), _slice("485000", None, "0.30")),
             NON_RESIDENT_AUTHORITY),
)


class ScheduleSet:
    """Finds the schedule in force, and names the failure when there is none.

    Because every schedule carries an end date, a missing one can be reported
    precisely: before the first, after the last, or in a gap between two.
    """

    def __init__(self, schedules: Sequence[Schedule] = SCHEDULES) -> None:
        if not schedules:
            raise ValueError("a schedule set needs at least one schedule")
        self.schedules = tuple(schedules)

    def __len__(self) -> int:
        return len(self.schedules)

    def __iter__(self) -> Iterator[Schedule]:
        return iter(self.schedules)

    def __repr__(self) -> str:
        return f"ScheduleSet({len(self)} schedules)"

    def spans(self, residency: str) -> list[str]:
        return [s.span for s in self.schedules if s.residency == residency]

    def in_force(self, when: dt.date, residency: str) -> Schedule:
        """The schedule governing when."""
        residency = Residency.check(residency)
        relevant = [s for s in self.schedules if s.residency == residency]
        if not relevant:
            raise NoScheduleForDateError(when, residency,
                                         sorted({s.residency for s in self.schedules}))
        for schedule in relevant:
            if schedule.covers(when):
                return schedule

        earlier = [s for s in relevant if s.closes is not None and s.closes < when]
        later = [s for s in relevant if s.opens > when]
        raise NoScheduleForDateError(when, residency, self.spans(residency),
                                     in_gap=bool(earlier and later))


class PayeTax(Tax):
    """Pay As You Earn on monthly employment income."""

    name = "PAYE"

    def __init__(self, schedules: ScheduleSet | None = None) -> None:
        self.schedules = schedules or ScheduleSet()

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """PAYE for one month.

        Anything already deducted before PAYE, which here means Local Service
        Tax, arrives through settings['deducted_first']. This class never needs
        to know what LST is.
        """
        schedule = self.schedules.in_force(ctx.period, ctx.residency)
        base = ctx.pay
        taken = ctx.settings.get("deducted_first")
        if taken is not None:
            base = base - Money(taken)
            if base.is_negative():
                base = Money.none()

        payable, working = schedule.charge(base)
        notes = [schedule.span, schedule.authority]
        if taken is not None:
            notes.insert(0, f"{Money(taken)} deducted before PAYE")
        return TaxResult(tax=self.name, charged_on=base, payable=payable,
                         entries=working, notes=tuple(notes))
