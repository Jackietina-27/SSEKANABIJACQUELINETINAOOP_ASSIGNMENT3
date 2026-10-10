# ==[ markdown ]==
# # Mini-Project 2: Uganda Tax Engine
#
# **What this is.** A testable calculator for the taxes URA administers —
# PAYE, NSSF, Local Service Tax, VAT and corporate income tax — with an
# engagement planner as the extension.
#
# **How this build is arranged.** `src/tax/money.py` holds the `Money` value
# object; `core.py` the abstract `Tax` contract and the errors; `paye.py` the
# progressive slices and effective-dated schedules; `ledger.py` the payroll as
# a set of **postings**; `vat.py`, `business.py` and `planner.py` the rest.
#
# **Two choices shape everything below.** Money **truncates towards zero**
# rather than rounding, so the engine never charges a shilling the computation
# did not fully earn — and the notebook *measures* what that costs instead of
# asserting it is fine. And payroll lines are postings that record **what they
# were charged on**, because "5% of what?" is the question an auditor asks and
# the answer differs between lines.
#
# **Rates.** The baseline table in the brief, cross-checked against the Income
# Tax Act Cap 340 Third Schedule as published by URA. Each schedule carries
# its authority, printed below.

# ==[ code ]==
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import datetime as dt
import random
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
import pandas as pd

from src.tax.business import (BusinessIncomeTax, CompanyTax, Outgoing, PaymentKind,
                              WITHHOLDING_RATES, WithholdingTax)
from src.tax.core import (BadRate, InvalidIncomeError, NoScheduleForDateError,
                          Residency, Tax, TaxContext, TaxError, TaxResult,
                          UnknownSupply)
from src.tax.ledger import Payroll, lst_monthly
from src.tax.money import Money, float_evidence, rounding_bias
from src.tax.paye import PayeTax, Schedule, ScheduleSet, SCHEDULES
from src.tax.planner import Comparison, EngagementPlanner
from src.tax.vat import Document, SupplyKind, VatTax
from src.weather.display import INK, house_style, save

house_style()
pd.set_option("display.width", 240)

PERIOD = dt.date(2026, 7, 1)
print(f"calculating for the month beginning {PERIOD.isoformat()}")

# ==[ markdown ]==
# ## The schedules and their authority

# ==[ code ]==
for schedule in SCHEDULES:
    print(schedule.span)
    print(schedule.to_frame().to_string(index=False))
    print(f"  plus {schedule.extra_rate:.0%} above {schedule.extra_above}")
    print(f"  authority: {schedule.authority}\n")

# ==[ markdown ]==
# ## Why `float` cannot be used, and what rounding costs
#
# Three verified failures, then a measurement of this build's own policy.

# ==[ code ]==
evidence = float_evidence()

print("1. ROUNDING GOES THE WRONG WAY")
print(f"   round(2.675, 2)  -> {evidence['rounded_float']}")
print(f"   exact answer     -> {evidence['rounded_exact']}")
print(f"   because 2.675 is really stored as:")
print(f"     {evidence['really_stored_as']}")
print("   The stored double is BELOW 2.675, so half-up rounds it DOWN. A tax")
print("   figure wrong by a shilling, with nothing in the decimal digits to")
print("   explain it.")

print("\n2. ADDITION IS NOT ASSOCIATIVE")
print(f"   (0.1+0.2)+0.3 = {evidence['left_grouping']!r}")
print(f"   0.1+(0.2+0.3) = {evidence['right_grouping']!r}")
print(f"   equal? {evidence['associative']}")
print("   A payslip sums LST, PAYE and NSSF. In float the total would depend")
print("   on the order, so two faithful implementations could disagree.")

print("\n3. ERROR ACCUMULATES")
print(f"   10,000 applications of a rate drift by {evidence['drift']} UGX")
print(f"   ({evidence['classic']}, since {evidence['classic_value']})")

# ==[ code ]==
print("Money refuses float at the door:")
for call, label in ((lambda: Money(335000.50), "Money(335000.50)"),
                    (lambda: Money("1000") * 0.3, "Money * 0.3"),
                    (lambda: Money("1000") * Money("2"), "Money * Money")):
    try:
        call()
        print(f"  {label:<20} -> allowed (it should not be)")
    except TypeError as problem:
        print(f"  {label:<20} -> TypeError: {str(problem)[:62]}")

print("\nAnd it truncates towards zero rather than rounding:")
for value in ("2.9", "0.99", "2.5", "-2.9"):
    half_up = Decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    half_even = Decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    print(f"  {value:>6}  truncated {str(Money(value).amount):>4}   "
          f"half-up {str(half_up):>4}   half-even {str(half_even):>4}")

# ==[ markdown ]==
# ### What truncation costs
#
# Truncation never charges a shilling the computation did not earn, which is
# the reason to choose it. It is not free, and the honest thing is to measure
# the cost rather than assert it is negligible. The cell below computes the
# **exact, unrounded** PAYE for 2,000 random salaries and totals it three ways.

# ==[ code ]==
random.seed(2026)
schedule = SCHEDULES[0]
exact_charges = []
for salary in random.sample(range(300_000, 15_000_000), 2000):
    gross = Money(salary)
    running = Decimal(0)
    for band in schedule:
        running += Decimal(band.rate) * band.portion_of(gross).amount
    if gross > schedule.extra_above:
        running += schedule.extra_rate * (gross - schedule.extra_above).amount
    exact_charges.append(running)

bias = rounding_bias(exact_charges)
pd.Series({k: str(v) for k, v in bias.items()}).to_frame("value")

# ==[ code ]==
print(f"over {bias['payslips']:,} payslips:")
print(f"  truncation collects {-bias['truncation_vs_half_up']} UGX less than "
      f"half-up")
print(f"  and {-bias['truncation_vs_half_even']} UGX less than half-to-even")
print(f"  the shortfall against the unrounded total is "
      f"{bias['truncation_shortfall']} UGX")
print()
print("Under a shilling per payslip — but one-sided, and it scales with the")
print("payroll. That is the point worth making: a rounding mode is a policy")
print("decision with a measurable revenue consequence, invisible in any single")
print("calculation and perfectly visible in the aggregate. This build states")
print("the policy and measures it rather than leaving it implicit.")

# ==[ markdown ]==
# ## 1. PAYE calculator
#
# ### The band edges the brief names

# ==[ code ]==
paye = PayeTax()
edges = [335_000, 335_001, 410_000, 485_000, 10_000_000, 10_000_001]
pd.DataFrame([{
    "monthly gross": g,
    "PAYE": int(paye.calculate(TaxContext(period=PERIOD,
                                          earnings=Money(g))).payable.amount),
    "effective rate": f"{paye.calculate(TaxContext(period=PERIOD, earnings=Money(g))).rate:.4%}",
} for g in edges]).set_index("monthly gross")

# ==[ markdown ]==
# Two of those look wrong and are right. **335,001 charges nothing** — 20% of
# one shilling is 0.2, which truncates away. **10,000,001 equals 10,000,000**
# for the same reason. What matters at that threshold is the marginal rate.

# ==[ code ]==
at = paye.calculate(TaxContext(period=PERIOD, earnings=Money(10_000_000))).payable
over = paye.calculate(TaxContext(period=PERIOD, earnings=Money(11_000_000))).payable
print(f"at 10,000,000        : {at}")
print(f"at 11,000,000        : {over}")
print(f"extra on the million : {over - at}")
print(f"marginal rate        : {(over - at).amount / 1_000_000:.0%}")
print("  = the 30% band, which continues, plus the 10% additional charge.")
assert (over - at) == Money("400000")

# ==[ code ]==
print(paye.calculate(TaxContext(period=PERIOD,
                                earnings=Money(12_000_000))).working())

# ==[ markdown ]==
# ### Resident against non-resident

# ==[ code ]==
levels = [200_000, 335_000, 500_000, 1_000_000, 3_000_000, 12_000_000]
table = pd.DataFrame({
    "resident": [int(paye.calculate(TaxContext(
        period=PERIOD, earnings=Money(g))).payable.amount) for g in levels],
    "non-resident": [int(paye.calculate(TaxContext(
        period=PERIOD, earnings=Money(g),
        residency=Residency.NON_RESIDENT)).payable.amount) for g in levels],
}, index=levels)
table.index.name = "monthly gross"
table["extra"] = table["non-resident"] - table["resident"]
table

# ==[ code ]==
grid = list(range(0, 15_000_001, 50_000))
curves = {status: [float(paye.calculate(TaxContext(
    period=PERIOD, earnings=Money(g), residency=status)).rate) for g in grid]
    for status in (Residency.RESIDENT, Residency.NON_RESIDENT)}

figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for index, (status, series) in enumerate(curves.items()):
    axes[0].plot([g / 1e6 for g in grid], [r * 100 for r in series],
                 color=INK[index], lw=1.8, label=status)
axes[0].axvline(10, color="0.4", ls="--", lw=1, label="10m threshold")
axes[0].set_xlabel("monthly gross (UGX millions)")
axes[0].set_ylabel("effective rate (%)")
axes[0].set_title("Effective rate rises towards the margin, never reaching it")
axes[0].legend(fontsize=8)

marginal = [(curves[Residency.RESIDENT][i + 1] * grid[i + 1]
             - curves[Residency.RESIDENT][i] * grid[i]) / 50_000
            for i in range(len(grid) - 1)]
axes[1].step([g / 1e6 for g in grid[1:]], [m * 100 for m in marginal],
             color=INK[2], lw=1.6, where="pre")
axes[1].axvline(10, color="0.4", ls="--", lw=1)
axes[1].set_xlabel("monthly gross (UGX millions)")
axes[1].set_ylabel("marginal rate (%)")
axes[1].set_title("Marginal rate: 0, 20, 25, 30, then 40 above 10m")
axes[1].set_ylim(-2, 45)
figure.tight_layout()
print("saved:", save(figure, "mp2_paye.png").name)
plt.show()

# ==[ markdown ]==
# ### Schedules that know when they stop
#
# Every schedule carries an **end date** as well as a start. That is what lets
# the engine distinguish three different failures, where a "most recent
# schedule at or before this date" lookup would silently apply an expired
# schedule to a later year and return a confident wrong number.

# ==[ code ]==
try:
    paye.calculate(TaxContext(period=dt.date(2000, 1, 1),
                              earnings=Money("500000")))
except NoScheduleForDateError as problem:
    print(f"before everything -> in_gap={problem.in_gap}: {problem}")

gapped = ScheduleSet([
    Schedule(Residency.RESIDENT, dt.date(2020, 7, 1), dt.date(2021, 6, 30),
             SCHEDULES[0].slices, "a closed year"),
    Schedule(Residency.RESIDENT, dt.date(2023, 7, 1), None,
             SCHEDULES[0].slices, "the current year")])
print(f"\ninside a schedule -> "
      f"{gapped.in_force(dt.date(2020, 9, 1), Residency.RESIDENT).span}")
try:
    gapped.in_force(dt.date(2022, 1, 1), Residency.RESIDENT)
except NoScheduleForDateError as problem:
    print(f"in the gap        -> in_gap={problem.in_gap}: {problem}")

# ==[ markdown ]==
# ## 2. Payroll as a ledger
#
# ### The order, and the justification
#
# Local Service Tax is an allowable deduction against employment income, so
# PAYE is charged on gross **less** LST — the brief says "applied before PAYE"
# and the Local Governments Act agrees. NSSF is the step most often got wrong:
# the employee's 5% is charged on the **full gross**, and Uganda does *not*
# allow it as a deduction against employment income the way several
# neighbouring jurisdictions do, so PAYE and NSSF sit on overlapping bases and
# neither reduces the other. The employer's 10% never touches net pay.

# ==[ code ]==
payroll = Payroll()
payroll.order()

# ==[ code ]==
slip = payroll.run(TaxContext(period=PERIOD, earnings=Money("2500000"),
                              allowances=Money("300000")))
print(slip.working())

# ==[ code ]==
slip.to_frame()

# ==[ markdown ]==
# ### Why postings instead of subtractions
#
# Each line records the base it was computed from. That makes the overlapping
# bases visible in the data rather than implicit in the order of statements.

# ==[ code ]==
for posting in slip:
    print(" ", posting)
print()
print(f"PAYE was charged on {slip.line('PAYE').charged_on}, which is gross")
print(f"less the {slip.line('Local Service Tax').amount} of LST.")
print(f"NSSF was charged on {slip.line('NSSF employee 5%').charged_on}, the")
print("full gross — the LST deduction does not reduce it. Two different bases")
print("in the same payslip, which is exactly the thing worth making explicit.")

# ==[ code ]==
no_lst = Payroll(lst=False).run(TaxContext(period=PERIOD,
                                           earnings=Money("2800000")))
print(f"PAYE with LST first : {slip.paye.payable}")
print(f"PAYE without LST    : {no_lst.paye.payable}")
print(f"difference          : {no_lst.paye.payable - slip.paye.payable}")
print(f"LST itself          : {slip.line('Local Service Tax').amount}")
print("\nAt a 30% marginal rate, deducting LST first returns 30% of it through")
print("a smaller PAYE charge. Getting the order wrong overstates the tax.")

# ==[ code ]==
ladder = []
for gross in (500_000, 1_000_000, 2_000_000, 5_000_000, 12_000_000):
    row = payroll.run(TaxContext(period=PERIOD, earnings=Money(gross)))
    ladder.append({"gross": gross,
                   "LST": int(row.line("Local Service Tax").amount.amount),
                   "PAYE": int(row.paye.payable.amount),
                   "NSSF 5%": int(row.line("NSSF employee 5%").amount.amount),
                   "net": int(row.net.amount),
                   "take-home": f"{row.take_home:.1%}",
                   "employer cost": int(row.employer_cost.amount)})
pd.DataFrame(ladder).set_index("gross")

# ==[ markdown ]==
# ## 3. VAT
#
# ### Zero-rated is not exempt
#
# Both charge no output VAT, so a sales report cannot tell them apart. A
# **zero-rated** supply is taxable at 0%, so input VAT incurred making it stays
# recoverable. An **exempt** supply is outside the VAT system, so the input VAT
# attributable to it is blocked and becomes a cost. Collapsing the two always
# overstates the refund.

# ==[ code ]==
pd.DataFrame([{"supply": str(kind), "output rate": f"{kind.output_rate:.0%}",
               "input VAT recoverable": kind.recovers_input}
              for kind in SupplyKind])

# ==[ code ]==
documents = [
    Document("INV-001 local sales", Money("10000000"), SupplyKind.STANDARD),
    Document("INV-002 coffee export", Money("4000000"), SupplyKind.ZERO_RATED),
    Document("INV-003 financial services", Money("3000000"), SupplyKind.EXEMPT),
    Document("PUR-010 packaging", Money("5000000"), SupplyKind.STANDARD,
             inbound=True),
    Document("PUR-011 export freight", Money("2000000"), SupplyKind.ZERO_RATED,
             inbound=True, vat_shown=Money("360000")),
    Document("PUR-012 branch costs", Money("1500000"), SupplyKind.EXEMPT,
             inbound=True, vat_shown=Money("270000")),
]
for document in documents:
    print(f"  {document}")

position = VatTax().position(documents)
print()
print(position.working())

# ==[ code ]==
position.by_kind

# ==[ code ]==
print("If the exempt supplies were wrongly treated as zero-rated:")
print(f"  the blocked {position.blocked} would have been reclaimed,")
print(f"  turning {position.net} payable into "
      f"{position.net - position.blocked}.")
print("  An understatement of the liability — and what an audit looks for.")

# ==[ markdown ]==
# ## 4. Corporate income tax

# ==[ code ]==
costs = [Outgoing("staff costs", Money("40000000")),
         Outgoing("rent", Money("12000000")),
         Outgoing("accounting depreciation", Money("8000000"), allowable=False),
         Outgoing("URA penalties", Money("2000000"), allowable=False)]
for cost in costs:
    print(f"  {cost}")

company = CompanyTax()
result = company.calculate(TaxContext(period=PERIOD, receipts=Money("100000000"),
                                      settings={"outgoings": costs}))
print()
print(result.working())

# ==[ code ]==
book = Money("100000000")
for cost in costs:
    book = book - cost.amount
print(f"accounting profit : {book}")
print(f"chargeable income : {result.charged_on}")
print(f"added back        : {result.charged_on - book}")

loss = company.calculate(TaxContext(period=PERIOD, receipts=Money("10000000"),
                                    outgoings=Money("25000000")))
print(f"\na loss charges nothing: base {loss.charged_on}, due {loss.payable}")

# ==[ markdown ]==
# ## Extension: employment or consultancy?
#
# A company needs someone's work. It can put them on the payroll, or engage
# them as an independent consultant. The two are taxed very differently, and
# the comparison only means anything if the **company's outlay is held
# constant** — otherwise one route simply buys more labour.
#
# * **Employment.** A salary of `S` costs `S` plus 10% employer NSSF, so an
#   outlay of `C` supports `S = C / 1.1`. The worker bears LST, PAYE and 5%
#   employee NSSF.
# * **Consultancy.** The whole outlay is paid as fees. 6% is withheld at
#   source, and the consultant is assessed on fees less business expenses at
#   the same progressive individual rates — with the withholding a **credit**
#   against that liability, not an extra charge.

# ==[ code ]==
planner = EngagementPlanner(PERIOD)
print(repr(planner))
print()
result = planner.compare(Money("5000000"))
print(result.employment.working())
print()
print(result.consultancy.working())

# ==[ code ]==
result.to_frame()

# ==[ code ]==
print(f"better on cash in hand : {result.better_for_cash}, "
      f"by {result.cash_margin}")
print(f"better on cash + NSSF  : {result.better_for_value}, "
      f"by {result.value_margin}")
print(f"counting NSSF closes   : {result.nssf_closes} of the cash gap")

# ==[ code ]==
sweep = planner.sweep([Money(str(v)) for v in
                       (1_000_000, 2_000_000, 5_000_000, 8_000_000,
                        12_000_000, 20_000_000)])
sweep

# ==[ markdown ]==
# ### Testing the obvious explanation, and finding it wrong
#
# The natural guess is that the consultant's expense allowance drives the
# result. It does not. The sweep below holds everything else constant and
# varies the allowance from zero upwards.

# ==[ code ]==
sensitivity = planner.sensitivity(
    Money("5000000"),
    [Decimal(v) for v in ("0.00", "0.10", "0.20", "0.30", "0.40", "0.50")])
sensitivity

# ==[ code ]==
print("Consultancy keeps more cash at EVERY allowance, including zero.")
print("So the allowance moves the margin but not the ranking, and the real")
print("driver is something else: employment carries 15% NSSF (5% employee")
print("plus 10% employer) that consultancy does not.")
print()
print("But NSSF is not a tax. It is the worker's own money, held for later.")
print("Counting it as value rather than cost is what the 'cash + saving'")
print(f"column does, and it closes {result.nssf_closes} of a "
      f"{result.cash_margin} gap —")
print("about two thirds of it. The cash comparison alone is misleading.")

# ==[ code ]==
outlays = list(range(1_000_000, 20_000_001, 500_000))
employee_cash, employee_value, consultant_cash = [], [], []
for amount in outlays:
    row = planner.compare(Money(str(amount)))
    employee_cash.append(int(row.employment.in_hand.amount))
    employee_value.append(int(row.employment.total_value.amount))
    consultant_cash.append(int(row.consultancy.in_hand.amount))

figure, ax = plt.subplots(figsize=(10, 5))
ax.plot([o / 1e6 for o in outlays], [v / 1e6 for v in consultant_cash],
        color=INK[1], lw=1.8, label="consultant: cash")
ax.plot([o / 1e6 for o in outlays], [v / 1e6 for v in employee_value],
        color=INK[2], lw=1.8, label="employee: cash + NSSF")
ax.plot([o / 1e6 for o in outlays], [v / 1e6 for v in employee_cash],
        color=INK[0], lw=1.8, ls="--", label="employee: cash only")
ax.fill_between([o / 1e6 for o in outlays],
                [v / 1e6 for v in employee_cash],
                [v / 1e6 for v in employee_value],
                color=INK[2], alpha=0.15, label="NSSF, saved not spent")
ax.set_xlabel("monthly cost to the company (UGX millions)")
ax.set_ylabel("value to the worker (UGX millions)")
ax.set_title("Employment against consultancy, at equal cost to the engager")
ax.legend(fontsize=8)
figure.tight_layout()
print("saved:", save(figure, "mp2_engagement.png").name)
plt.show()

# ==[ markdown ]==
# ### What must be said alongside any of this

# ==[ code ]==
for index, caveat in enumerate(Comparison.caveats(), 1):
    print(f"{index}. {caveat}\n")

# ==[ markdown ]==
# The first caveat is the most important and is not a technicality. Whether a
# relationship **is** employment or consultancy is a question of fact, decided
# on the substance of the arrangement — who controls the work, how integrated
# the person is, who carries the risk — and not on the label the parties find
# convenient. URA can and does re-characterise arrangements, with the tax and
# penalties falling on the engager. This notebook answers the narrow question
# "what would each cost if both were genuinely available", which is not the
# same as "which should we choose".

# ==[ markdown ]==
# ## One contract, five taxes

# ==[ code ]==
shared = TaxContext(period=PERIOD, earnings=Money("3000000"),
                    receipts=Money("50000000"), outgoings=Money("20000000"),
                    documents=tuple(documents))
pd.DataFrame([{
    "tax": outcome.tax, "charged on": int(outcome.charged_on.amount),
    "payable": int(outcome.payable.amount), "rate": f"{outcome.rate:.2%}",
} for outcome in (tax.calculate(shared) for tax in
                  (PayeTax(), VatTax(), CompanyTax(), WithholdingTax(),
                   BusinessIncomeTax()))]).set_index("tax")

# ==[ markdown ]==
# ## Validation: nothing fails silently

# ==[ code ]==
for label, attempt in [
    ("negative pay", lambda: TaxContext(period=PERIOD, earnings=Money("-1"))),
    ("unknown residency", lambda: TaxContext(period=PERIOD,
                                             earnings=Money("1"),
                                             residency="lunar")),
    ("a float VAT rate", lambda: VatTax(rate=0.18)),
    ("an unknown supply", lambda: Document("X", Money("100"), kind="standard")),
    ("a negative cost", lambda: Outgoing("odd", Money("-1"))),
    ("an uncovered date", lambda: PayeTax().calculate(
        TaxContext(period=dt.date(1999, 1, 1), earnings=Money("500000")))),
    ("a float amount", lambda: Money(1.5)),
    ("an impossible allowance", lambda: BusinessIncomeTax(
        expense_allowance=Decimal("1.5"))),
]:
    try:
        attempt()
        print(f"  {label:<24} -> NOT CAUGHT")
    except (TaxError, ValueError, TypeError) as problem:
        print(f"  {label:<24} -> {type(problem).__name__}: "
              f"{str(problem)[:52]}")

# ==[ markdown ]==
# ## Discussion
#
# ### 1. Why is `float` unsafe for tax computation?
#
# Because binary floating point cannot represent most decimal fractions, and
# tax is defined in decimal. The three demonstrations above are distinct
# failures. The first would reach a payslip directly: `round(2.675, 2)` returns
# `2.67`, because the nearest double to 2.675 lies fractionally below it. A
# round-half-up rule applied to a float therefore rounds half *down* whenever
# the stored value falls on the low side, and nothing in the decimal digits
# predicts which way it will go. The outcome is a figure wrong by one shilling,
# inconsistently, with no diagnosable cause — the worst available failure mode
# in a system whose output somebody is legally obliged to pay.
#
# The second is specific to payroll. Float addition is not associative, so a
# payslip summing LST, PAYE and NSSF would produce a total that depends on the
# order the lines were added. Two faithful implementations of the same
# documented rules could disagree, and no reading of the rules would settle
# which was correct. The third, accumulation, is worst in exactly the
# high-volume runs where it is least likely to be noticed.
#
# `Decimal` fixes all three by storing decimal digits and letting the rounding
# rule be stated and not inherited from the hardware. This build states it
# as **truncation towards zero**, which never charges a shilling the
# computation did not fully earn. Truncation is not neutral, which is why the
# notebook measures it: across 2,000 payslips it collected under a shilling per
# payslip less than half-up — small, one-sided, and scaling with the payroll.
# Stating a rounding policy and then quantifying its bias seems to me the
# minimum standard for a calculation of this kind, because the bias is
# invisible in any single result and perfectly visible in the aggregate.
#
# `Money` refuses `float` at construction and refuses float rates in
# multiplication. Refusing rather than coercing matters: `Money(335000.50)`
# could silently become `Money('335000')`, and a system that quietly discards
# half a shilling is worse than one that stops and says why.
#
# ### 2. Where is inheritance justified, and where was it deliberately avoided?
#
# **Justified: `Tax`.** `PayeTax`, `VatTax`, `CompanyTax`, `WithholdingTax` and
# `BusinessIncomeTax` all subclass it. What they share is not behaviour but a
# *contract* — `calculate(ctx) -> TaxResult` — and the base class deliberately
# holds no arithmetic, because the five do mechanically unrelated things. The
# payoff is visible in the summary table above, which loops over all five, and
# in the planner, which accepts any `Tax` without knowing which it has.
#
# **Deliberately avoided: `BusinessIncomeTax` does not inherit from
# `PayeTax`.** This was the closest call in the project. A consultant is
# charged at *exactly* the same progressive rates as an employee, so
# `class BusinessIncomeTax(PayeTax)` would have reused the band machinery and
# looked natural. It would also have been wrong. A consultant is not an
# employee: the base is fees less business expenses instead of gross pay,
# the assessment is annual rather than monthly, no NSSF applies, no Local
# Service Tax applies, and the withholding credit has no analogue in PAYE.
# Subclassing would have inherited `calculate` and then had to override
# essentially all of it, which is the clearest possible signal that the
# relationship is not "is a". So the class **composes** a `PayeTax` internally
# and delegates the band arithmetic to it. Reuse without the false claim of
# kinship — and if PAYE later gains an employment-specific rule, the consultant
# calculation does not silently acquire it.
#
# **Deliberately avoided: `Money` does not subclass `Decimal`.** Inheriting
# would have made the arithmetic work for free, and would also have meant
# `Money * Money` quietly returning a bare `Decimal`, `Money + 0.5` admitting a
# float through an inherited operator, and the truncation policy being
# bypassable by any inherited method. The purpose of the type is its ability to
# *refuse*, and a subclass cannot refuse what its parent permits.
#
# **Deliberately avoided: `Recorder` does not subclass the live caller** in
# mini-project 1. They share an interface and no implementation at all. A
# subclass would inherit a session, a schedule and a timeout the recorder has
# no use for, and would invite someone to call an inherited method that makes a
# real request from an object whose entire purpose is not to.
#
# The test running through all four: inherit where the subtypes owe a caller
# the same promise; compose where they merely happen to need similar code.

# ==[ markdown ]==
# ## Testing
#
# `tests/test_tax.py` covers the band edges the brief lists and the
# property-based checks it asks for with `hypothesis` — tax non-negative,
# monotonic in income, never exceeding income. The cell below re-runs the
# three inline over a wide sample.

# ==[ code ]==
random.seed(2026)
sample = sorted(random.sample(range(0, 80_000_000), 500) + edges)
charges = [paye.calculate(TaxContext(period=PERIOD,
                                     earnings=Money(g))).payable for g in sample]

non_negative = all(not c.is_negative() for c in charges)
within = all(c <= Money(g) for c, g in zip(charges, sample))
rising = all(b >= a for a, b in zip(charges, charges[1:]))

print(f"over {len(sample)} incomes from 0 to 80,000,000:")
print(f"  never negative       : {non_negative}")
print(f"  never exceeds income : {within}")
print(f"  monotonic            : {rising}")
assert non_negative and within and rising
print("\nhypothesis checks the same three over generated inputs in")
print("tests/test_tax.py, which is stricter than this fixed sample.")
