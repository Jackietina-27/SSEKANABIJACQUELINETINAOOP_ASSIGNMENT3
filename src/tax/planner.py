"""Extension: employment against consultancy, at equal cost to the engager.

A company needs someone's work and can either put them on the payroll or
engage them as an independent consultant. The two are taxed very differently,
and the comparison only means anything if the company's total outlay is held
constant, since otherwise one route simply buys more labour than the other.

- Employment. A salary of S costs the company S plus 10% employer NSSF, so an
  outlay of C supports S = C / 1.1. The employee bears Local Service Tax,
  PAYE and 5% employee NSSF.
- Consultancy. The whole outlay C is paid as fees, with 6% withheld at
  source. The consultant is assessed on fees less business expenses at the
  same progressive individual rates, and the withholding is a credit against
  that liability, not an extra charge.

What drives the answer is not the consultant's expense allowance, which was
the obvious guess and is wrong: even at a zero allowance the consultancy
route keeps more cash, because employment carries 15% NSSF (5% employee plus
10% employer) that consultancy does not. But NSSF is the worker's own money
held for later, not a tax, so comparing cash alone understates employment
badly. total_value counts cash plus saving, and that is the comparison to
read.

One caveat belongs with any use of this. Whether a relationship is employment
or consultancy is a question of fact, decided on the substance of the
arrangement: control, integration, who bears risk. It is not a label the
parties may choose for tax reasons, and URA can and does re-characterise
arrangements. This module answers what each would cost if both were genuinely
available, which is narrower than which one to pick, and caveats() says so in
the output.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator

import pandas as pd

from src.tax.business import (BusinessIncomeTax, DEFAULT_EXPENSE_ALLOWANCE,
                              PaymentKind, WithholdingTax)
from src.tax.core import Residency, TaxContext
from src.tax.ledger import EMPLOYER_RATE, Payroll
from src.tax.money import Money

__all__ = ["Arrangement", "Comparison", "EngagementPlanner"]


@dataclass(frozen=True)
class Arrangement:
    """What one engagement route costs and leaves behind."""

    name: str
    outlay: Money
    received_before_tax: Money
    charges: tuple[tuple[str, Money], ...]
    in_hand: Money
    retirement_saving: Money = None

    def __iter__(self) -> Iterator[tuple[str, Money]]:
        return iter(self.charges)

    @property
    def total_tax(self) -> Money:
        return Money.total(amount for _, amount in self.charges)

    @property
    def tax_rate(self) -> Decimal:
        """Total tax as a share of the company's outlay."""
        return self.total_tax.fraction_of(self.outlay)

    @property
    def kept(self) -> Decimal:
        """Share of the outlay reaching the worker as cash."""
        return self.in_hand.fraction_of(self.outlay)

    @property
    def total_value(self) -> Money:
        """Cash plus retirement saving.

        NSSF is withheld from the employee and matched by the employer, so on a
        cash measure it looks like pure cost. It is the employee's money, held
        for later, so cash alone understates employment by a wide margin.
        """
        saving = self.retirement_saving or Money.none()
        return self.in_hand + saving

    @property
    def value_kept(self) -> Decimal:
        """Share of outlay reaching the worker as cash or saving."""
        return self.total_value.fraction_of(self.outlay)

    def __repr__(self) -> str:
        return (f"Arrangement({self.name}, in hand={self.in_hand}, "
                f"tax={self.total_tax}, {self.tax_rate:.1%})")

    def working(self) -> str:
        lines = [f"{self.name}: the company spends {self.outlay} a month",
                 f"    {'worker is paid':<30} {self.received_before_tax:>18}"]
        lines += [f"    {label:<30} {amount:>18}" for label, amount in self.charges]
        lines += [f"    {'total tax and contributions':<30} "
                  f"{self.total_tax:>18}",
                  f"    {'worker keeps':<30} {self.in_hand:>18}",
                  f"    ({self.tax_rate:.2%} of outlay to tax; "
                  f"{self.kept:.2%} kept as cash)"]
        if self.retirement_saving and not self.retirement_saving.is_none():
            lines.append(f"    plus {self.retirement_saving} into NSSF, which is "
                         f"a saving rather than a cost")
        return "\n".join(lines)


@dataclass(frozen=True)
class Comparison:
    """Both arrangements at one level of company outlay."""

    employment: Arrangement
    consultancy: Arrangement

    @property
    def better_for_cash(self) -> str:
        return ("Consultancy" if self.consultancy.in_hand > self.employment.in_hand
                else "Employment")

    @property
    def cash_margin(self) -> Money:
        return abs(self.consultancy.in_hand - self.employment.in_hand)

    @property
    def better_for_value(self) -> str:
        """Which wins once NSSF is counted as the worker's money, not a cost."""
        return ("Consultancy"
                if self.consultancy.total_value > self.employment.total_value
                else "Employment")

    @property
    def value_margin(self) -> Money:
        return abs(self.consultancy.total_value - self.employment.total_value)

    @property
    def nssf_closes(self) -> Money:
        """How much of the cash gap counting NSSF closes."""
        return self.cash_margin - self.value_margin

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "arrangement": a.name,
            "cash in hand": int(a.in_hand.amount),
            "NSSF saved": int(a.retirement_saving.amount)
            if a.retirement_saving else 0,
            "cash + saving": int(a.total_value.amount),
            "total tax": int(a.total_tax.amount),
            "tax as % of outlay": f"{a.tax_rate:.2%}",
        } for a in (self.employment, self.consultancy)]).set_index("arrangement")

    @staticmethod
    def caveats() -> list[str]:
        """What a reader must know before acting on any of this."""
        return [
            "Whether a relationship is employment or consultancy is a question "
            "of fact — control, integration, who bears the risk — decided on "
            "substance, not a label chosen for tax reasons. URA can "
            "re-characterise an arrangement.",
            "The consultant's expense allowance is illustrative, not "
            "statutory; a real consultant deducts actual allowable costs. It "
            "shifts the margin but not the ranking — consultancy keeps more "
            "cash even at a zero allowance.",
            "Employment NSSF is a retirement asset, not a pure cost, and it "
            "is the single biggest driver of the gap here. Comparing cash in "
            "hand alone understates employment substantially; read the "
            "cash-plus-saving column instead.",
            "A consultant above the VAT registration threshold must register "
            "and charge VAT, which is not modelled.",
            "Employment carries rights — leave, notice, injury cover — with "
            "real value that no tax calculation captures.",
        ]


class EngagementPlanner:
    """Compares employment against consultancy at equal cost to the engager."""

    def __init__(self, period: dt.date, payroll: Payroll | None = None,
                 business: BusinessIncomeTax | None = None,
                 withholding: WithholdingTax | None = None,
                 residency: str = Residency.RESIDENT) -> None:
        self.period = period
        self.payroll = payroll or Payroll()
        self.business = business or BusinessIncomeTax()
        self.withholding = withholding or WithholdingTax(
            PaymentKind.PROFESSIONAL_FEE)
        self.residency = Residency.check(residency)

    def __repr__(self) -> str:
        return (f"EngagementPlanner({self.period.isoformat()}, "
                f"allowance={self.business.expense_allowance:.0%})")

    def as_employee(self, outlay: Money) -> Arrangement:
        """Put the worker on the payroll, spending outlay in total."""
        if outlay.is_negative():
            raise ValueError("outlay cannot be negative")
        salary = outlay * (Decimal(1) / (Decimal(1) + EMPLOYER_RATE))
        slip = self.payroll.run(TaxContext(period=self.period, earnings=salary,
                                           residency=self.residency))
        charges = tuple((posting.line, posting.amount) for posting in slip)
        employer_nssf = slip.line("NSSF employer 10%").amount
        employee_nssf = slip.line("NSSF employee 5%").amount
        return Arrangement("Employment", outlay, salary, charges, slip.net,
                           retirement_saving=employer_nssf + employee_nssf)

    def as_consultant(self, outlay: Money,
                      months: int = 12) -> Arrangement:
        """Engage the worker on fees, spending outlay in total.

        The whole outlay becomes fees, with no employer contribution, so the
        consultant is paid more gross than an employee would be. The
        withholding is deducted at source and then credited, so the net
        liability is the assessment less what was already withheld.
        """
        if outlay.is_negative():
            raise ValueError("outlay cannot be negative")
        annual_fees = outlay * Decimal(months)
        withheld = self.withholding.calculate(
            TaxContext(period=self.period, receipts=annual_fees)).payable
        assessment = self.business.calculate(
            TaxContext(period=self.period, receipts=annual_fees,
                       residency=self.residency))

        # Withholding is a credit: the balance is the assessment less it.
        balance = assessment.payable - withheld
        if balance.is_negative():
            balance = Money.none()

        monthly_tax = assessment.payable * (Decimal(1) / Decimal(months))
        charges = (("income tax on business income", monthly_tax),)
        in_hand = outlay - monthly_tax
        return Arrangement("Consultancy", outlay, outlay, charges, in_hand,
                           retirement_saving=Money.none())

    def compare(self, outlay: Money) -> Comparison:
        """Both routes at one level of outlay."""
        return Comparison(self.as_employee(outlay), self.as_consultant(outlay))

    def sweep(self, outlays: list[Money]) -> pd.DataFrame:
        """Both routes across a range of monthly outlays."""
        rows = []
        for outlay in outlays:
            result = self.compare(outlay)
            rows.append({
                "monthly outlay": int(outlay.amount),
                "employee keeps": int(result.employment.in_hand.amount),
                "consultant keeps": int(result.consultancy.in_hand.amount),
                "better on cash": result.better_for_cash,
                "cash margin": int(result.cash_margin.amount),
                "employee NSSF": int(result.employment.retirement_saving.amount),
                "employee cash+NSSF":
                    int(result.employment.total_value.amount),
                "better on total value": result.better_for_value,
                "value margin": int(result.value_margin.amount),
            })
        return pd.DataFrame(rows).set_index("monthly outlay")

    def sensitivity(self, outlay: Money,
                    allowances: list[Decimal]) -> pd.DataFrame:
        """How the answer moves with the consultant's expense allowance.

        The allowance is an assumption, not a statutory figure, so the
        comparison should show how much depends on it.
        """
        rows = []
        for allowance in allowances:
            planner = EngagementPlanner(
                self.period, self.payroll,
                BusinessIncomeTax(expense_allowance=allowance),
                self.withholding, self.residency)
            result = planner.compare(outlay)
            rows.append({
                "expense allowance": f"{allowance:.0%}",
                "consultant keeps": int(result.consultancy.in_hand.amount),
                "employee keeps": int(result.employment.in_hand.amount),
                "better for cash": result.better_for_cash,
            })
        return pd.DataFrame(rows).set_index("expense allowance")
