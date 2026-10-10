"""Corporate income tax, withholding, and tax on business income.

Corporate income tax is 30% of chargeable income: receipts less allowable
outgoings. A disallowed cost still reduces accounting profit but not taxable
profit, so the two diverge and the difference is added back where a reader
can see it.

BusinessIncomeTax exists because the extension compares employment with
consultancy. A consultant's fees are business income, charged at the same
progressive individual rates as employment but annually, with no NSSF, and
after an allowance for genuine business costs.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Sequence

import pandas as pd

from src.tax.core import (BadRate, Entry, InvalidIncomeError, Residency, Tax,
                          TaxContext, TaxResult)
from src.tax.money import Money
from src.tax.paye import PayeTax, ScheduleSet

__all__ = ["COMPANY_RATE", "BusinessIncomeTax", "CompanyTax", "Outgoing",
           "PaymentKind", "WITHHOLDING_RATES", "WithholdingTax"]

COMPANY_RATE = Decimal("0.30")


@dataclass(frozen=True)
class Outgoing:
    """A cost, flagged by whether the Act allows it as a deduction."""

    description: str
    amount: Money
    allowable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Money):
            object.__setattr__(self, "amount", Money(self.amount))
        if self.amount.is_negative():
            raise InvalidIncomeError(self.amount, "a cost cannot be negative")

    def __str__(self) -> str:
        return (f"{self.description}: {self.amount} "
                f"({'allowable' if self.allowable else 'DISALLOWED'})")


class CompanyTax(Tax):
    """Corporate income tax at a flat rate on chargeable income."""

    name = "Corporate income tax"

    def __init__(self, rate: Decimal = COMPANY_RATE) -> None:
        if not isinstance(rate, Decimal):
            raise BadRate(rate, "the company rate must be a Decimal")
        if not Decimal("0") <= rate <= Decimal("1"):
            raise BadRate(rate)
        self.rate = rate

    @staticmethod
    def chargeable(receipts: Money,
                   outgoings: Sequence[Outgoing]) -> tuple[Money, Money]:
        """Chargeable income, and the disallowed total added back."""
        allowed = Money.total(o.amount for o in outgoings if o.allowable)
        refused = Money.total(o.amount for o in outgoings if not o.allowable)
        income = receipts - allowed
        return (Money.none() if income.is_negative() else income), refused

    def calculate(self, ctx: TaxContext) -> TaxResult:
        detailed = ctx.settings.get("outgoings")
        if detailed:
            base, refused = self.chargeable(ctx.receipts, detailed)
            note = (f"{refused} disallowed and added back" if refused
                    else "every cost allowable")
        else:
            base = ctx.receipts - ctx.outgoings
            if base.is_negative():
                base = Money.none()
            refused, note = Money.none(), "outgoings taken as allowable"

        payable = base * self.rate
        notes = [note]
        if base.is_none():
            notes.append("a loss carries forward; nothing is charged this period")
        return TaxResult(tax=self.name, charged_on=base, payable=payable,
                         entries=(Entry("receipts less allowable outgoings", base,
                                        self.rate, payable),),
                         notes=tuple(notes))


class PaymentKind(Enum):
    """Payments that attract withholding, with their resident rates."""

    PROFESSIONAL_FEE = "professional fee"
    GOODS_AND_SERVICES = "goods and services"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    ROYALTY = "royalty"

    def __str__(self) -> str:
        return self.value


# Resident withholding rates (Income Tax Act Cap 340, Part VIII). Non-resident
# and treaty rates differ and are out of scope; the engine refuses instead of
# guessing.
WITHHOLDING_RATES: dict[PaymentKind, Decimal] = {
    PaymentKind.PROFESSIONAL_FEE: Decimal("0.06"),
    PaymentKind.GOODS_AND_SERVICES: Decimal("0.06"),
    PaymentKind.DIVIDEND: Decimal("0.15"),
    PaymentKind.INTEREST: Decimal("0.15"),
    PaymentKind.ROYALTY: Decimal("0.15"),
}

# Withholding on goods and services bites only above this threshold.
SUPPLY_THRESHOLD = Money("1000000")


class WithholdingTax(Tax):
    """Tax withheld at source on a payment.

    Withholding is a credit, not an extra charge: it is deducted at source and
    set against the recipient's final liability. Reporting it as additional tax
    would double-count it.
    """

    name = "Withholding tax"

    def __init__(self, kind: PaymentKind = PaymentKind.PROFESSIONAL_FEE,
                 rates: dict[PaymentKind, Decimal] | None = None,
                 threshold: Money = SUPPLY_THRESHOLD) -> None:
        if not isinstance(kind, PaymentKind):
            raise ValueError(f"unknown payment kind {kind!r}")
        self.kind = kind
        self.rates = dict(rates or WITHHOLDING_RATES)
        self.threshold = threshold

    @property
    def rate(self) -> Decimal:
        return self.rates[self.kind]

    def calculate(self, ctx: TaxContext) -> TaxResult:
        gross = ctx.receipts
        applies = (self.kind is not PaymentKind.GOODS_AND_SERVICES
                   or gross >= self.threshold)
        payable = gross * self.rate if applies else Money.none()
        notes = [f"{self.kind} withheld at {self.rate:.0%} (resident rate)",
                 "withholding is a credit against the final liability, "
                 "not an additional charge"]
        if not applies:
            notes.insert(1, f"below the {self.threshold} threshold; "
                            f"nothing withheld")
        notes.append("non-resident and treaty rates differ and are out of scope")
        return TaxResult(tax=self.name, charged_on=gross, payable=payable,
                         entries=(Entry(f"withholding on {self.kind}", gross,
                                        self.rate if applies else Decimal("0"),
                                        payable),),
                         notes=tuple(notes))


# Flat allowance for a consultant's business costs, used when actual costs are
# not supplied. This is an ILLUSTRATIVE figure, not a statutory one: a real
# consultant deducts actual allowable expenses. It is the single most
# consequential assumption in the extension, which is why it is a parameter.
DEFAULT_EXPENSE_ALLOWANCE = Decimal("0.30")


class BusinessIncomeTax(Tax):
    """Income tax on an individual's business or consultancy income.

    Business income is charged at the same progressive individual rates as
    employment income, so this class delegates to PayeTax instead of
    reimplementing the bands. Composition, not inheritance: a consultant is not
    an employee and none of this is PAYE. What differs is the base, which is
    fees less business expenses, assessed annually, with no NSSF and no Local
    Service Tax.
    """

    name = "Business income tax"

    def __init__(self, expense_allowance: Decimal = DEFAULT_EXPENSE_ALLOWANCE,
                 schedules: ScheduleSet | None = None, months: int = 12) -> None:
        if not isinstance(expense_allowance, Decimal):
            raise BadRate(expense_allowance, "the allowance must be a Decimal")
        if not Decimal("0") <= expense_allowance < Decimal("1"):
            raise BadRate(expense_allowance, "the allowance must lie in 0..1")
        if months < 1:
            raise ValueError("months must be at least 1")
        self.expense_allowance = expense_allowance
        self.months = int(months)
        self._bands = PayeTax(schedules)

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """Charge on annual fees in ctx.receipts.

        Actual expenses can be supplied in settings['expenses']; otherwise the
        flat allowance applies.
        """
        fees = ctx.receipts
        if fees.is_negative():
            raise InvalidIncomeError(fees, "fees cannot be negative")

        stated = ctx.settings.get("expenses")
        expenses = (Money(stated) if stated is not None
                    else fees * self.expense_allowance)
        if expenses > fees:
            expenses = fees
        taxable = fees - expenses

        monthly = taxable * (Decimal(1) / Decimal(self.months))
        per_month = self._bands.calculate(
            ctx.revised(earnings=monthly, allowances=Money.none(),
                        settings={}))
        payable = per_month.payable * Decimal(self.months)

        entries = (Entry("business expenses", fees,
                         self.expense_allowance if stated is None
                         else expenses.fraction_of(fees), expenses),
                   Entry(f"individual rates on {monthly} a month",
                         taxable, per_month.rate, payable))
        notes = [
            f"assessed on {taxable} of taxable business income",
            ("flat expense allowance of "
             f"{self.expense_allowance:.0%} (illustrative, not statutory)")
            if stated is None else "actual expenses supplied",
            "no NSSF and no Local Service Tax apply to business income",
        ]
        return TaxResult(tax=self.name, charged_on=taxable, payable=payable,
                         entries=entries, notes=tuple(notes))
