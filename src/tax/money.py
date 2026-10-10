"""Money in Ugandan shillings, exact and deliberately conservative.

Every amount here wraps a Decimal. float is refused outright, and
float_evidence() demonstrates three ways it gets money wrong.

This build truncates towards zero to whole shillings. The shilling has no
circulating subdivision, so a fraction is not payable, and of the ways to
remove it truncation is the only one that never charges a shilling the
computation did not fully earn: half-up and half-to-even both sometimes round
a liability up.

The choice is not free. rounding_bias() measures what it costs, because
across a large payroll truncation collects systematically less than either
rounding rule. A rounding mode is a policy decision with a measurable revenue
consequence, so the engine states the bias instead of leaving it implicit.
"""
from __future__ import annotations

from decimal import (ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal,
                     InvalidOperation, localcontext)
from typing import Any, Iterable, Union

__all__ = ["CURRENCY", "Money", "ROUNDING", "UNIT", "float_evidence",
           "rounding_bias"]

CURRENCY = "UGX"

# Whole shillings.
UNIT = Decimal("1")

# Truncate towards zero: never charge a shilling not fully earned.
ROUNDING = ROUND_DOWN

Amount = Union[int, str, Decimal, "Money"]


class Money:
    """An exact number of Ugandan shillings, truncated to whole units.

    >>> Money("335000") + Money(1)
    Money('335001')
    >>> Money("1000") * Decimal("0.205")      # 205 exactly
    Money('205')
    >>> Money("2.9")                          # truncated, not rounded
    Money('2')
    """

    __slots__ = ("_amount",)

    def __init__(self, value: Amount = 0) -> None:
        self._amount = self._parse(value)

    @staticmethod
    def _parse(value: Amount) -> Decimal:
        if isinstance(value, Money):
            return value._amount
        if isinstance(value, bool):
            raise TypeError("Money does not accept a bool")
        if isinstance(value, float):
            raise TypeError(
                "Money does not accept a float: round(2.675, 2) gives 2.67, "
                "because the nearest double sits below 2.675. Use an int, a "
                "str or a Decimal.")
        try:
            return Decimal(value).quantize(UNIT, rounding=ROUNDING)
        except (InvalidOperation, ValueError, TypeError) as problem:
            raise ValueError(f"{value!r} is not a readable amount") from problem

    @classmethod
    def none(cls) -> "Money":
        """Zero, named so running totals read clearly."""
        return cls(0)

    @classmethod
    def total(cls, amounts: Iterable["Money"]) -> "Money":
        """Sum a series of amounts, starting from zero."""
        running = cls(0)
        for amount in amounts:
            running = running + amount
        return running

    @property
    def amount(self) -> Decimal:
        return self._amount

    def is_none(self) -> bool:
        return self._amount == 0

    def is_negative(self) -> bool:
        return self._amount < 0

    def fraction_of(self, whole: "Money") -> Decimal:
        """This amount as a share of whole; zero when whole is zero."""
        other = Money(whole)
        if other.is_none():
            return Decimal("0")
        return (self._amount / other._amount).quantize(Decimal("0.000001"))

    def __add__(self, other: Amount) -> "Money":
        return Money(self._amount + Money._parse(other))

    __radd__ = __add__

    def __sub__(self, other: Amount) -> "Money":
        return Money(self._amount - Money._parse(other))

    def __rsub__(self, other: Amount) -> "Money":
        return Money(Money._parse(other) - self._amount)

    def __mul__(self, rate: int | str | Decimal) -> "Money":
        """Multiply by a rate. Money times Money is refused as meaningless.

        The product is formed at extended precision and truncated once, so a
        chain of rates does not accumulate intermediate error.
        """
        if isinstance(rate, Money):
            raise TypeError("shillings multiplied by shillings has no meaning")
        if isinstance(rate, float):
            raise TypeError("a rate must be a Decimal, not a float")
        with localcontext() as context:
            context.prec = 28
            return Money(Decimal(rate) * self._amount)

    __rmul__ = __mul__

    def __neg__(self) -> "Money":
        return Money(-self._amount)

    def __abs__(self) -> "Money":
        return Money(abs(self._amount))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (Money, int, str, Decimal)):
            try:
                return self._amount == Money._parse(other)
            except (TypeError, ValueError):
                return NotImplemented
        return NotImplemented

    def __lt__(self, other: Amount) -> bool:
        return self._amount < Money._parse(other)

    def __le__(self, other: Amount) -> bool:
        return self._amount <= Money._parse(other)

    def __gt__(self, other: Amount) -> bool:
        return self._amount > Money._parse(other)

    def __ge__(self, other: Amount) -> bool:
        return self._amount >= Money._parse(other)

    def __hash__(self) -> int:
        return hash((CURRENCY, self._amount))

    def __bool__(self) -> bool:
        return self._amount != 0

    def __repr__(self) -> str:
        return f"Money('{self._amount}')"

    def __str__(self) -> str:
        return f"{CURRENCY} {self._amount:,}"

    def __format__(self, spec: str) -> str:
        return format(self._amount, spec) if spec else str(self)


def rounding_bias(charges: Iterable[Decimal]) -> dict[str, Any]:
    """Measure what the rounding policy costs, across many charges.

    Takes the exact tax figures for a set of payslips and reports the total
    collected under truncation, half-up and half-to-even. The gaps are the
    revenue consequence of a decision that is invisible in any single
    calculation.
    """
    exact = [Decimal(charge) for charge in charges]
    if not exact:
        raise ValueError("there are no charges to measure")
    truncated = sum(c.quantize(UNIT, rounding=ROUND_DOWN) for c in exact)
    half_up = sum(c.quantize(UNIT, rounding=ROUND_HALF_UP) for c in exact)
    half_even = sum(c.quantize(UNIT, rounding=ROUND_HALF_EVEN) for c in exact)
    unrounded = sum(exact)
    return {
        "payslips": len(exact),
        "unrounded": unrounded,
        "truncated": truncated,
        "half_up": half_up,
        "half_even": half_even,
        "truncation_shortfall": unrounded - truncated,
        "truncation_vs_half_up": truncated - half_up,
        "truncation_vs_half_even": truncated - half_even,
    }


def float_evidence() -> dict[str, Any]:
    """Three verified ways binary floating point gets money wrong.

    1. round(2.675, 2) returns 2.67: the nearest double to 2.675 lies below
       it, so a half-up rule rounds half down, and unpredictably.
    2. Addition is not associative, so a payslip's total depends on the
       order the lines were added.
    3. Error accumulates, and grows with the number of operations.
    """
    binary, exact = 0.0, Decimal(0)
    for _ in range(10_000):
        binary += 335_001 * 0.2
        exact += Decimal(335_001) * Decimal("0.2")
    left, right = (0.1 + 0.2) + 0.3, 0.1 + (0.2 + 0.3)
    return {
        "rounded_float": round(2.675, 2),
        "rounded_exact": Decimal("2.675").quantize(Decimal("0.01"),
                                                   rounding=ROUND_HALF_UP),
        "rounding_is_wrong": round(2.675, 2) == 2.67,
        "really_stored_as": Decimal(2.675),
        "left_grouping": left,
        "right_grouping": right,
        "associative": left == right,
        "float_sum": binary,
        "exact_sum": exact,
        "drift": Decimal(repr(binary)) - exact,
        "classic": f"0.1 + 0.2 == 0.3 -> {0.1 + 0.2 == 0.3}",
        "classic_value": repr(0.1 + 0.2),
    }
