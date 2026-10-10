"""Payroll as a ledger of postings.

Order of deductions, and why:

1. gross = earnings + taxable allowances
2. Local Service Tax is posted first, because LST is an allowable deduction
   against employment income, so PAYE is charged on what remains. The brief
   says "applied before PAYE" and the Local Governments Act agrees.
3. PAYE on gross less LST
4. NSSF employee, 5% of gross: charged on the full gross, and not deductible
   against PAYE. This is the step most often got wrong. Uganda does not allow
   the employee contribution as a deduction from employment income the way
   several neighbouring jurisdictions do, so PAYE and NSSF are charged on
   overlapping bases and neither reduces the other.
5. NSSF employer, 10%, is posted with against_employee=False. It is a cost of
   employment borne on top of the wage and never reduces net pay.

As postings rather than a chain of subtractions, each line records what it
was charged on and whether it affects the PAYE base and the employee's net.
Net pay falls out of summing the ledger, and "what was this line charged on"
is answered by the data instead of by reading the order of statements.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator, Sequence

import pandas as pd

from src.tax.core import BadRate, Tax, TaxContext, TaxResult
from src.tax.money import Money
from src.tax.paye import PayeTax

__all__ = ["EMPLOYEE_RATE", "EMPLOYER_RATE", "LST_SCALE", "Payroll", "Posting",
           "Payslip", "lst_monthly"]

EMPLOYEE_RATE = Decimal("0.05")
EMPLOYER_RATE = Decimal("0.10")

# Indicative annual Local Service Tax by monthly income, collected in four
# instalments. LST is set by each local government, so this is a documented
# default and the payroll accepts an override.
LST_SCALE: tuple[tuple[Money, Money], ...] = (
    (Money("100000"), Money("5000")),
    (Money("200000"), Money("10000")),
    (Money("300000"), Money("20000")),
    (Money("400000"), Money("30000")),
    (Money("500000"), Money("40000")),
    (Money("1000000"), Money("60000")),
    (Money("99999999999"), Money("100000")),
)


def lst_monthly(gross: Money, scale: Sequence[tuple[Money, Money]] = LST_SCALE,
                instalments: int = 4) -> Money:
    """The monthly LST instalment implied by a gross figure."""
    if gross.is_negative():
        raise ValueError("gross cannot be negative")
    if instalments < 1:
        raise ValueError("instalments must be at least 1")
    for ceiling, annual in scale:
        if gross <= ceiling:
            return annual * (Decimal(1) / Decimal(instalments))
    return scale[-1][1] * (Decimal(1) / Decimal(instalments))


@dataclass(frozen=True)
class Posting:
    """One line in the payroll ledger."""

    line: str
    amount: Money
    charged_on: Money
    before_paye: bool = False
    against_employee: bool = True

    @property
    def effective_rate(self) -> Decimal:
        return self.amount.fraction_of(self.charged_on)

    def __str__(self) -> str:
        who = "employee" if self.against_employee else "employer"
        return (f"{self.line}: {self.amount} on {self.charged_on} "
                f"({self.effective_rate:.2%}, borne by the {who})")


@dataclass(frozen=True)
class Payslip:
    """One month's ledger, and what falls out of it."""

    period: dt.date
    gross: Money
    postings: tuple[Posting, ...]
    paye: TaxResult

    def __iter__(self) -> Iterator[Posting]:
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    def __repr__(self) -> str:
        return (f"Payslip({self.period.isoformat()}, gross={self.gross}, "
                f"net={self.net})")

    @property
    def employee_deductions(self) -> Money:
        return Money.total(p.amount for p in self.postings if p.against_employee)

    @property
    def employer_cost(self) -> Money:
        """Gross plus everything the employer bears on top of it."""
        borne = Money.total(p.amount for p in self.postings
                            if not p.against_employee)
        return self.gross + borne

    @property
    def net(self) -> Money:
        return self.gross - self.employee_deductions

    @property
    def take_home(self) -> Decimal:
        return self.net.fraction_of(self.gross)

    def line(self, name: str) -> Posting:
        """One posting by name."""
        for posting in self.postings:
            if posting.line == name:
                return posting
        raise KeyError(f"no posting called {name!r}; "
                       f"have {[p.line for p in self.postings]}")

    def to_frame(self) -> pd.DataFrame:
        rows = [{"line": "Gross pay", "amount": int(self.gross.amount),
                 "charged on": "—", "before PAYE": "—", "borne by": "—"}]
        rows += [{"line": p.line,
                  "amount": -int(p.amount.amount) if p.against_employee
                  else int(p.amount.amount),
                  "charged on": int(p.charged_on.amount),
                  "before PAYE": "yes" if p.before_paye else "no",
                  "borne by": "employee" if p.against_employee else "employer"}
                 for p in self.postings]
        rows.append({"line": "Net pay", "amount": int(self.net.amount),
                     "charged on": "—", "before PAYE": "—", "borne by": "—"})
        return pd.DataFrame(rows)

    def working(self) -> str:
        out = [f"Payroll for {self.period.isoformat()}",
               f"  Gross pay                     {self.gross:>18}"]
        for posting in self.postings:
            if posting.against_employee:
                out.append(f"  less {posting.line:<24} {-posting.amount:>18}")
        out += [f"  Net pay                       {self.net:>18}",
                f"  (take-home {self.take_home:.2%})"]
        for posting in self.postings:
            if not posting.against_employee:
                out.append(f"  {posting.line + ' (employer)':<30} "
                           f"{posting.amount:>18}")
        out.append(f"  Total cost of employment      {self.employer_cost:>18}")
        return "\n".join(out)


class Payroll:
    """Builds the ledger, in the documented order."""

    def __init__(self, paye: Tax | None = None, lst: bool = True,
                 lst_amount: Money | None = None,
                 employee_rate: Decimal = EMPLOYEE_RATE,
                 employer_rate: Decimal = EMPLOYER_RATE) -> None:
        for rate in (employee_rate, employer_rate):
            if not isinstance(rate, Decimal):
                raise BadRate(rate, "NSSF rates must be Decimal")
            if not Decimal("0") <= rate <= Decimal("1"):
                raise BadRate(rate)
        self.paye = paye or PayeTax()
        self.lst = bool(lst)
        self.lst_amount = lst_amount
        self.employee_rate = employee_rate
        self.employer_rate = employer_rate

    def __repr__(self) -> str:
        return (f"Payroll(lst={'on' if self.lst else 'off'}, "
                f"employee={self.employee_rate:.0%}, "
                f"employer={self.employer_rate:.0%})")

    def order(self) -> pd.DataFrame:
        """The documented order as data, so it can be read and justified."""
        return pd.DataFrame([
            {"#": 1, "line": "Local Service Tax", "charged on": "gross",
             "before PAYE": "yes", "borne by": "employee"},
            {"#": 2, "line": "PAYE", "charged on": "gross less LST",
             "before PAYE": "—", "borne by": "employee"},
            {"#": 3, "line": "NSSF employee 5%", "charged on": "full gross",
             "before PAYE": "no", "borne by": "employee"},
            {"#": 4, "line": "NSSF employer 10%", "charged on": "full gross",
             "before PAYE": "no", "borne by": "employer"},
        ])

    def run(self, ctx: TaxContext) -> Payslip:
        """Post every line and return the completed payslip."""
        gross = ctx.pay
        postings: list[Posting] = []

        local = Money.none()
        if self.lst:
            local = (Money(self.lst_amount) if self.lst_amount is not None
                     else lst_monthly(gross))
        postings.append(Posting("Local Service Tax", local, gross,
                                before_paye=True))

        paye = self.paye.calculate(
            ctx.revised(settings={**dict(ctx.settings), "deducted_first": local}))
        postings.append(Posting("PAYE", paye.payable, paye.charged_on))

        postings.append(Posting("NSSF employee 5%", gross * self.employee_rate,
                                gross))
        postings.append(Posting("NSSF employer 10%", gross * self.employer_rate,
                                gross, against_employee=False))

        return Payslip(period=ctx.period, gross=gross,
                       postings=tuple(postings), paye=paye)
