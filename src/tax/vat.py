"""VAT: standard-rated, zero-rated and exempt supplies.

Zero-rated and exempt both charge no output VAT, so a sales report cannot
tell them apart. They are not the same. A zero-rated supply is taxable at 0%,
so input VAT incurred making it stays recoverable. An exempt supply is
outside the VAT system, so input VAT attributable to it is blocked and
becomes a cost of doing business. Collapsing the two always overstates the
refund, so VatPosition reports blocked input separately instead of netting it
away.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Sequence

import pandas as pd

from src.tax.core import BadRate, Entry, Tax, TaxContext, TaxResult, UnknownSupply
from src.tax.money import Money

__all__ = ["Document", "STANDARD", "SupplyKind", "VatPosition", "VatTax"]

STANDARD = Decimal("0.18")


class SupplyKind(Enum):
    """How a supply is treated for VAT."""

    STANDARD = "standard-rated"
    ZERO_RATED = "zero-rated"
    EXEMPT = "exempt"

    @property
    def output_rate(self) -> Decimal:
        return STANDARD if self is SupplyKind.STANDARD else Decimal("0")

    @property
    def recovers_input(self) -> bool:
        """True for standard and zero-rated supplies, False for exempt ones."""
        return self is not SupplyKind.EXEMPT

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Document:
    """A sale or a purchase."""

    reference: str
    net: Money
    kind: SupplyKind = SupplyKind.STANDARD
    inbound: bool = False
    vat_shown: Money | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SupplyKind):
            raise UnknownSupply(self.kind)
        if not isinstance(self.net, Money):
            object.__setattr__(self, "net", Money(self.net))
        if self.net.is_negative():
            raise ValueError(f"{self.reference}: a net amount cannot be negative")

    @property
    def vat(self) -> Money:
        if self.vat_shown is not None:
            return Money(self.vat_shown)
        return self.net * self.kind.output_rate

    @property
    def gross(self) -> Money:
        return self.net + self.vat

    def __str__(self) -> str:
        side = "purchase" if self.inbound else "sale"
        return f"{self.reference} ({side}, {self.kind}): {self.net} + {self.vat} VAT"


@dataclass(frozen=True)
class VatPosition:
    """A completed VAT position for a period."""

    output: Money
    recoverable: Money
    blocked: Money
    by_kind: pd.DataFrame = None

    @property
    def net(self) -> Money:
        """Positive when payable, negative when a refund is due."""
        return self.output - self.recoverable

    @property
    def is_refund(self) -> bool:
        return self.net.is_negative()

    def __repr__(self) -> str:
        which = "refund" if self.is_refund else "payable"
        return (f"VatPosition(output={self.output}, "
                f"recoverable={self.recoverable}, {which}={abs(self.net)})")

    def working(self) -> str:
        label = "refundable" if self.is_refund else "payable"
        return "\n".join([
            f"  Output VAT on sales            {self.output:>18}",
            f"  less recoverable input VAT     {-self.recoverable:>18}",
            f"  = net VAT {label:<20}{abs(self.net):>18}",
            f"  input VAT blocked (exempt)     {self.blocked:>18}",
        ])


class VatTax(Tax):
    """Value Added Tax over a set of documents."""

    name = "VAT"

    def __init__(self, rate: Decimal = STANDARD) -> None:
        if not isinstance(rate, Decimal):
            raise BadRate(rate, "the VAT rate must be a Decimal")
        if not Decimal("0") <= rate <= Decimal("1"):
            raise BadRate(rate)
        self.rate = rate

    @staticmethod
    def output_vat(documents: Sequence[Document]) -> Money:
        return Money.total(d.vat for d in documents if not d.inbound)

    @staticmethod
    def input_vat(documents: Sequence[Document]) -> tuple[Money, Money]:
        """Input VAT as (recoverable, blocked)."""
        recoverable = Money.total(d.vat for d in documents
                                  if d.inbound and d.kind.recovers_input)
        blocked = Money.total(d.vat for d in documents
                              if d.inbound and not d.kind.recovers_input)
        return recoverable, blocked

    def position(self, documents: Sequence[Document]) -> VatPosition:
        """The full position, with a breakdown by supply kind."""
        recoverable, blocked = self.input_vat(documents)
        rows = []
        for kind in SupplyKind:
            sales = [d for d in documents if not d.inbound and d.kind is kind]
            buys = [d for d in documents if d.inbound and d.kind is kind]
            rows.append({
                "supply": str(kind),
                "net sales": int(Money.total(d.net for d in sales).amount),
                "output VAT": int(Money.total(d.vat for d in sales).amount),
                "input VAT": int(Money.total(d.vat for d in buys).amount),
                "input recoverable": kind.recovers_input,
            })
        return VatPosition(output=self.output_vat(documents),
                           recoverable=recoverable, blocked=blocked,
                           by_kind=pd.DataFrame(rows))

    def calculate(self, ctx: TaxContext) -> TaxResult:
        """The Tax contract, over ctx.documents."""
        documents = tuple(ctx.documents)
        position = self.position(documents)
        standard_sales = Money.total(
            d.net for d in documents
            if not d.inbound and d.kind is SupplyKind.STANDARD)

        entries = (Entry("output VAT on standard-rated sales", standard_sales,
                         self.rate, position.output),
                   Entry("recoverable input VAT", position.recoverable,
                         Decimal("-1"), -position.recoverable))
        notes = [f"{position.blocked} of input VAT blocked as attributable to "
                 f"exempt supplies"]
        if position.is_refund:
            notes.append("this is a refund position, not a payment")
        # A refund is negative, which TaxResult forbids; the magnitude is
        # reported and the direction recorded in the notes.
        return TaxResult(tax=self.name, charged_on=standard_sales,
                         payable=abs(position.net), entries=entries,
                         notes=tuple(notes))
