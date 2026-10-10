"""The shared tax abstraction and the error hierarchy.

Tax declares calculate(ctx) -> TaxResult and holds no arithmetic. The taxes
here do mechanically unrelated things, so there is nothing to inherit. What
the base supplies is a common shape, which is what lets a ledger, a report or
a planner treat any of them alike.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterator, Mapping

from src.tax.money import Money

__all__ = ["BadRate", "Entry", "InvalidIncomeError", "NoScheduleForDateError",
           "Residency", "Tax", "TaxContext", "TaxError", "TaxResult",
           "UnknownSupply"]


class TaxError(Exception):
    """Base class for every error this engine raises."""


class InvalidIncomeError(TaxError):
    """An amount cannot serve as a chargeable income."""

    def __init__(self, amount: object, why: str = "income cannot be negative"):
        self.amount, self.why = amount, why
        super().__init__(f"{why}: {amount}")


class BadRate(TaxError):
    """A rate is not a Decimal, or lies outside 0..1."""

    def __init__(self, rate: object, why: str = "a rate must lie in 0..1"):
        self.rate, self.why = rate, why
        super().__init__(f"{why}: {rate}")


class NoScheduleForDateError(TaxError):
    """No schedule covers the date being calculated.

    Raised both when a date precedes every schedule and when it falls in a gap
    between two. The second is only detectable because schedules carry an end
    date; a "latest schedule at or before this date" lookup would apply an
    expired one instead.
    """

    def __init__(self, when: dt.date, who: str = "",
                 on_file: list[str] | None = None,
                 in_gap: bool = False) -> None:
        self.when, self.who, self.on_file, self.in_gap = when, who, on_file or [], in_gap
        whom = f" for a {who}" if who else ""
        where = "falls in a gap between schedules" if in_gap else "is not covered"
        listing = f"; on file: {', '.join(self.on_file)}" if self.on_file else ""
        super().__init__(f"{when}{whom} {where}{listing}")


class UnknownSupply(TaxError):
    """An invoice carries a supply type VAT cannot rate."""

    def __init__(self, supply: object) -> None:
        self.supply = supply
        super().__init__(f"unknown supply type {supply!r}")


class Residency:
    """Residency status, validated on the way in."""

    RESIDENT = "resident"
    NON_RESIDENT = "non-resident"

    @classmethod
    def check(cls, value: str) -> str:
        text = str(value).strip().lower()
        if text not in {cls.RESIDENT, cls.NON_RESIDENT}:
            raise ValueError(
                f"residency must be 'resident' or 'non-resident', got {value!r}")
        return text


@dataclass(frozen=True)
class TaxContext:
    """Everything any tax might need, so the Tax signature stays uniform.

    Each tax ignores what does not concern it. A separate context per tax would
    make a generic ledger or planner impossible to write.
    """

    period: dt.date
    earnings: Money = field(default_factory=Money.none)
    allowances: Money = field(default_factory=Money.none)
    residency: str = Residency.RESIDENT
    receipts: Money = field(default_factory=Money.none)
    outgoings: Money = field(default_factory=Money.none)
    documents: tuple = ()
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.period, dt.date):
            raise TypeError("period must be a datetime.date")
        object.__setattr__(self, "residency", Residency.check(self.residency))
        for name in ("earnings", "allowances", "receipts", "outgoings"):
            value = getattr(self, name)
            if not isinstance(value, Money):
                object.__setattr__(self, name, Money(value))
            if getattr(self, name).is_negative():
                raise InvalidIncomeError(getattr(self, name),
                                         f"{name} cannot be negative")

    @property
    def pay(self) -> Money:
        """Earnings plus taxable allowances, before any deduction."""
        return self.earnings + self.allowances

    def revised(self, **changes: Any) -> "TaxContext":
        """A copy with some fields replaced; the context stays frozen."""
        current = {"period": self.period, "earnings": self.earnings,
                   "allowances": self.allowances, "residency": self.residency,
                   "receipts": self.receipts, "outgoings": self.outgoings,
                   "documents": self.documents, "settings": self.settings}
        current.update(changes)
        return TaxContext(**current)


@dataclass(frozen=True)
class Entry:
    """One line of the arithmetic behind a result."""

    description: str
    applied_to: Money
    rate: Decimal
    produces: Money

    def __str__(self) -> str:
        return f"{self.description}: {self.applied_to} @ {self.rate} = {self.produces}"


@dataclass(frozen=True)
class TaxResult:
    """What a tax computed, and how it got there."""

    tax: str
    charged_on: Money
    payable: Money
    entries: tuple[Entry, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.payable.is_negative():
            raise InvalidIncomeError(self.payable, "tax payable cannot be negative")

    @property
    def rate(self) -> Decimal:
        """Tax as a share of the base; zero when nothing is charged."""
        return self.payable.fraction_of(self.charged_on)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (f"TaxResult({self.tax}, on={self.charged_on}, "
                f"payable={self.payable}, {self.rate:.2%})")

    def working(self) -> str:
        """The full calculation, one line per entry."""
        out = [f"{self.tax} on {self.charged_on}"]
        out += [f"  {entry}" for entry in self.entries]
        out.append(f"  = {self.payable}   (effective {self.rate:.2%})")
        out += [f"  note: {note}" for note in self.notes]
        return "\n".join(out)


class Tax(ABC):
    """The contract every tax in this project keeps."""

    name: str = "tax"

    @abstractmethod
    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Compute this tax for one context."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
