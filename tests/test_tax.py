"""Tests for Mini-Project 2.

The band edges the brief names, the hypothesis properties it asks for, and
the behaviours this build's design choices imply — truncating rounding, gap
detection in the schedule set, and the postings ledger.
"""
import datetime as dt
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.tax.business import (BusinessIncomeTax, CompanyTax, Outgoing, PaymentKind,
                              WITHHOLDING_RATES, WithholdingTax)
from src.tax.core import (BadRate, InvalidIncomeError, NoScheduleForDateError,
                          Residency, Tax, TaxContext, TaxError, TaxResult,
                          UnknownSupply)
from src.tax.ledger import Payroll, Payslip, Posting, lst_monthly
from src.tax.money import Money, float_evidence, rounding_bias
from src.tax.paye import (PayeTax, Schedule, ScheduleSet, SCHEDULES, Slice)
from src.tax.planner import Comparison, EngagementPlanner
from src.tax.vat import Document, SupplyKind, VatTax

PERIOD = dt.date(2026, 7, 1)


def resident(pay) -> TaxContext:
    return TaxContext(period=PERIOD, earnings=Money(pay))


def visitor(pay) -> TaxContext:
    return TaxContext(period=PERIOD, earnings=Money(pay),
                      residency=Residency.NON_RESIDENT)


class TestMoney:
    def test_float_is_refused(self):
        with pytest.raises(TypeError, match="float"):
            Money(335000.50)
        with pytest.raises(TypeError):
            Money("1000") * 0.3
        with pytest.raises(TypeError):
            Money(True)

    def test_exact_inputs_agree(self):
        assert Money(335000) == Money("335000") == Money(Decimal("335000"))

    def test_nonsense_rejected(self):
        with pytest.raises(ValueError):
            Money("not an amount")

    def test_the_four_required_operators(self):
        a, b = Money("335000"), Money("75000")
        assert a + b == Money("410000")                  # __add__
        assert a * Decimal("0.2") == Money("67000")      # __mul__
        assert b < a                                     # __lt__
        assert repr(a) == "Money('335000')"              # __repr__

    def test_money_times_money_refused(self):
        with pytest.raises(TypeError, match="meaning"):
            Money("1000") * Money("2")

    def test_rounding_truncates_towards_zero(self):
        """This build never charges a shilling the computation did not earn."""
        assert Money("2.9") == Money("2")
        assert Money("0.99") == Money("0")
        assert Money("-2.9") == Money("-2")          # towards zero, not down
        assert Money("3") == Money("3")

    def test_truncation_differs_from_both_rounding_rules(self):
        value = Decimal("2.5")
        assert Money(value) == Money("2")
        assert value.quantize(Decimal("1"), rounding=ROUND_HALF_UP) == 3
        assert value.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN) == 2

    def test_total_sums_from_zero(self):
        assert Money.total([Money("1"), Money("2"), Money("3")]) == Money("6")
        assert Money.total([]) == Money.none()

    def test_fraction_of(self):
        assert Money("250").fraction_of(Money("1000")) == Decimal("0.250000")
        assert Money("5").fraction_of(Money.none()) == Decimal("0")

    def test_the_rounding_bias_is_measurable_and_one_sided(self):
        """Truncation collects less than either rounding rule. By how much?"""
        charges = [Decimal("100.7"), Decimal("200.5"), Decimal("300.2"),
                   Decimal("400.9")]
        bias = rounding_bias(charges)
        assert bias["truncated"] < bias["half_up"]
        assert bias["truncated"] <= bias["half_even"]
        assert bias["truncation_shortfall"] > 0
        assert bias["payslips"] == 4

    def test_rounding_bias_needs_data(self):
        with pytest.raises(ValueError):
            rounding_bias([])

    def test_the_float_failures_are_real(self):
        evidence = float_evidence()
        assert evidence["rounding_is_wrong"] is True
        assert evidence["rounded_float"] == 2.67
        assert evidence["rounded_exact"] == Decimal("2.68")
        assert evidence["associative"] is False
        assert evidence["drift"] != 0


class TestPayeBoundaries:
    """Every band edge the brief lists."""

    @pytest.mark.parametrize("pay,expected", [
        (335_000, "0"), (335_001, "0"), (410_000, "15000"),
        (485_000, "33750"), (10_000_000, "2888250"), (10_000_001, "2888250")])
    def test_resident_edges(self, pay, expected):
        assert PayeTax().calculate(resident(pay)).payable == Money(expected)

    def test_the_nil_band_charges_nothing(self):
        for pay in (0, 1, 200_000, 334_999, 335_000):
            assert PayeTax().calculate(resident(pay)).payable.is_none()

    def test_no_cliff_at_any_edge(self):
        paye = PayeTax()
        for edge in (335_000, 410_000, 485_000, 10_000_000):
            rise = (paye.calculate(resident(edge + 1)).payable
                    - paye.calculate(resident(edge)).payable)
            assert rise <= Money("1")

    def test_the_top_marginal_rate_is_forty_percent(self):
        paye = PayeTax()
        extra = (paye.calculate(resident(11_000_000)).payable
                 - paye.calculate(resident(10_000_000)).payable)
        assert extra == Money("400000")

    @pytest.mark.parametrize("pay,expected", [
        (100_000, "10000"), (335_000, "33500"),
        (410_000, "48500"), (485_000, "67250")])
    def test_non_resident_edges(self, pay, expected):
        assert PayeTax().calculate(visitor(pay)).payable == Money(expected)

    def test_non_residents_never_pay_less(self):
        paye = PayeTax()
        for pay in (50_000, 335_000, 1_000_000, 20_000_000):
            assert paye.calculate(visitor(pay)).payable >= \
                paye.calculate(resident(pay)).payable

    def test_the_nil_band_appears_in_the_working(self):
        """So a reader can see how much income fell in it."""
        result = PayeTax().calculate(resident(12_000_000))
        assert len(result.entries) == 5          # four slices plus the extra
        assert result.entries[0].produces.is_none()
        assert "additional charge" in " ".join(e.description for e in result)


class TestPayeProperties:
    """The three properties the brief asks hypothesis to check."""

    money = st.integers(min_value=0, max_value=200_000_000)

    @given(pay=money)
    @settings(max_examples=200, deadline=None)
    def test_never_negative(self, pay):
        assert not PayeTax().calculate(resident(pay)).payable.is_negative()

    @given(base=money, more=st.integers(min_value=0, max_value=5_000_000))
    @settings(max_examples=200, deadline=None)
    def test_monotonic(self, base, more):
        paye = PayeTax()
        assert paye.calculate(resident(base + more)).payable >= \
            paye.calculate(resident(base)).payable

    @given(pay=money)
    @settings(max_examples=200, deadline=None)
    def test_never_exceeds_income(self, pay):
        assert PayeTax().calculate(resident(pay)).payable <= Money(pay)

    @given(pay=st.integers(min_value=1, max_value=200_000_000))
    @settings(max_examples=100, deadline=None)
    def test_average_rate_below_the_margin(self, pay):
        assert PayeTax().calculate(resident(pay)).rate < Decimal("0.40")


class TestSchedules:
    def test_a_date_before_everything_raises(self):
        with pytest.raises(NoScheduleForDateError) as caught:
            PayeTax().calculate(TaxContext(period=dt.date(2000, 1, 1),
                                           earnings=Money("500000")))
        assert caught.value.in_gap is False

    def test_a_gap_between_schedules_is_named_as_such(self):
        """Only possible because schedules carry an end date."""
        gapped = ScheduleSet([
            Schedule(Residency.RESIDENT, dt.date(2020, 7, 1),
                     dt.date(2021, 6, 30), SCHEDULES[0].slices, "closed"),
            Schedule(Residency.RESIDENT, dt.date(2023, 7, 1), None,
                     SCHEDULES[0].slices, "current")])
        assert gapped.in_force(dt.date(2020, 9, 1),
                               Residency.RESIDENT).opens == dt.date(2020, 7, 1)
        with pytest.raises(NoScheduleForDateError) as in_gap:
            gapped.in_force(dt.date(2022, 1, 1), Residency.RESIDENT)
        assert in_gap.value.in_gap is True
        with pytest.raises(NoScheduleForDateError) as before:
            gapped.in_force(dt.date(2019, 1, 1), Residency.RESIDENT)
        assert before.value.in_gap is False

    def test_slices_must_be_contiguous_and_closed(self):
        with pytest.raises(ValueError, match="contiguous"):
            Schedule(Residency.RESIDENT, PERIOD, None,
                     (Slice(Money.none(), Money("100"), Decimal("0")),
                      Slice(Money("200"), None, Decimal("0.3"))))
        with pytest.raises(ValueError, match="open-ended"):
            Schedule(Residency.RESIDENT, PERIOD, None,
                     (Slice(Money.none(), Money("100"), Decimal("0")),))
        with pytest.raises(ValueError, match="start at zero"):
            Schedule(Residency.RESIDENT, PERIOD, None,
                     (Slice(Money("100"), None, Decimal("0.3")),))

    def test_a_schedule_cannot_close_before_it_opens(self):
        with pytest.raises(ValueError):
            Schedule(Residency.RESIDENT, dt.date(2021, 1, 1),
                     dt.date(2020, 1, 1), SCHEDULES[0].slices)

    @pytest.mark.parametrize("rate", [Decimal("-0.1"), Decimal("1.5"), 0.3, "0.3"])
    def test_slice_rates_must_be_sane_decimals(self, rate):
        with pytest.raises(BadRate):
            Slice(Money.none(), None, rate)

    def test_unknown_residency_rejected(self):
        with pytest.raises(ValueError):
            TaxContext(period=PERIOD, earnings=Money("1"), residency="lunar")

    def test_every_schedule_cites_its_authority(self):
        assert all(schedule.authority for schedule in SCHEDULES)


class TestLedger:
    def test_the_documented_order(self):
        order = Payroll().order()
        assert list(order["line"]) == ["Local Service Tax", "PAYE",
                                       "NSSF employee 5%", "NSSF employer 10%"]
        assert order.loc[0, "before PAYE"] == "yes"
        assert order.loc[3, "borne by"] == "employer"

    def test_every_posting_records_what_it_was_charged_on(self):
        """'5% of what?' is the question an auditor asks."""
        slip = Payroll().run(TaxContext(period=PERIOD, earnings=Money("2000000")))
        assert slip.line("NSSF employee 5%").charged_on == Money("2000000")
        # PAYE is charged on gross less LST, which is a different base
        assert slip.line("PAYE").charged_on < slip.gross

    def test_lst_reduces_the_paye_base(self):
        ctx = TaxContext(period=PERIOD, earnings=Money("2000000"))
        assert (Payroll().run(ctx).paye.payable
                < Payroll(lst=False).run(ctx).paye.payable)

    def test_nssf_is_charged_on_full_gross(self):
        slip = Payroll().run(TaxContext(period=PERIOD, earnings=Money("2000000")))
        assert slip.line("NSSF employee 5%").amount == \
            Money("2000000") * Decimal("0.05")
        assert slip.line("NSSF employer 10%").amount == \
            Money("2000000") * Decimal("0.10")

    def test_the_employer_posting_never_reduces_net(self):
        slip = Payroll().run(TaxContext(period=PERIOD, earnings=Money("1000000")))
        assert slip.line("NSSF employer 10%").against_employee is False
        assert slip.employer_cost > slip.gross
        assert slip.net == slip.gross - slip.employee_deductions

    def test_the_ledger_balances(self):
        slip = Payroll().run(TaxContext(period=PERIOD, earnings=Money("2500000"),
                                        allowances=Money("300000")))
        employee = Money.total(p.amount for p in slip if p.against_employee)
        assert slip.gross - employee == slip.net
        assert Decimal("0") < slip.take_home < Decimal("1")

    def test_lst_can_be_switched_off_or_fixed(self):
        ctx = TaxContext(period=PERIOD, earnings=Money("2000000"))
        assert Payroll(lst=False).run(ctx).line("Local Service Tax").amount.is_none()
        assert Payroll(lst_amount=Money("7500")).run(ctx).line(
            "Local Service Tax").amount == Money("7500")

    @pytest.mark.parametrize("gross,instalment", [
        ("50000", "1250"), ("250000", "5000"), ("2000000", "25000")])
    def test_lst_scale(self, gross, instalment):
        assert lst_monthly(Money(gross)) == Money(instalment)

    def test_an_unknown_posting_raises(self):
        slip = Payroll().run(TaxContext(period=PERIOD, earnings=Money("1000000")))
        with pytest.raises(KeyError):
            slip.line("Bonus")

    def test_nssf_rates_must_be_decimals(self):
        with pytest.raises(BadRate):
            Payroll(employee_rate=0.05)

    def test_negative_pay_refused(self):
        with pytest.raises(InvalidIncomeError):
            TaxContext(period=PERIOD, earnings=Money("-1"))


class TestVat:
    @pytest.fixture
    def documents(self):
        return [Document("S1", Money("10000000"), SupplyKind.STANDARD),
                Document("S2", Money("4000000"), SupplyKind.ZERO_RATED),
                Document("S3", Money("3000000"), SupplyKind.EXEMPT),
                Document("P1", Money("5000000"), SupplyKind.STANDARD, inbound=True),
                Document("P2", Money("2000000"), SupplyKind.ZERO_RATED,
                         inbound=True, vat_shown=Money("360000")),
                Document("P3", Money("1500000"), SupplyKind.EXEMPT,
                         inbound=True, vat_shown=Money("270000"))]

    def test_output_vat_on_standard_sales_only(self, documents):
        assert VatTax().output_vat(documents) == Money("1800000")

    def test_zero_and_exempt_charge_the_same_output(self):
        assert SupplyKind.ZERO_RATED.output_rate == \
            SupplyKind.EXEMPT.output_rate == Decimal("0")

    def test_but_only_exempt_blocks_recovery(self, documents):
        assert SupplyKind.ZERO_RATED.recovers_input is True
        assert SupplyKind.EXEMPT.recovers_input is False
        recoverable, blocked = VatTax().input_vat(documents)
        assert recoverable == Money("1260000") and blocked == Money("270000")

    def test_the_net_position(self, documents):
        position = VatTax().position(documents)
        assert position.net == Money("540000") and not position.is_refund

    def test_a_refund_is_reported_as_one(self):
        documents = [Document("S", Money("1000000"), SupplyKind.STANDARD),
                     Document("P", Money("9000000"), SupplyKind.STANDARD,
                              inbound=True)]
        assert VatTax().position(documents).is_refund
        result = VatTax().calculate(TaxContext(period=PERIOD,
                                               documents=tuple(documents)))
        assert not result.payable.is_negative()
        assert any("refund" in note for note in result.notes)

    def test_an_unknown_supply_rejected(self):
        with pytest.raises(UnknownSupply):
            Document("X", Money("100"), kind="standard")

    def test_a_negative_document_rejected(self):
        with pytest.raises(ValueError):
            Document("X", Money("-100"))


class TestCompanyAndWithholding:
    def test_flat_thirty_percent(self):
        result = CompanyTax().calculate(TaxContext(
            period=PERIOD, receipts=Money("100000000"),
            outgoings=Money("40000000")))
        assert result.charged_on == Money("60000000")
        assert result.payable == Money("18000000")

    def test_disallowed_costs_added_back(self):
        costs = [Outgoing("staff", Money("40000000")),
                 Outgoing("penalties", Money("2000000"), allowable=False)]
        result = CompanyTax().calculate(TaxContext(
            period=PERIOD, receipts=Money("100000000"),
            settings={"outgoings": costs}))
        assert result.charged_on == Money("60000000")
        assert "disallowed" in " ".join(result.notes)

    def test_a_loss_charges_nothing(self):
        result = CompanyTax().calculate(TaxContext(
            period=PERIOD, receipts=Money("10000000"),
            outgoings=Money("25000000")))
        assert result.charged_on.is_none() and result.payable.is_none()

    def test_a_negative_cost_rejected(self):
        with pytest.raises(InvalidIncomeError):
            Outgoing("odd", Money("-1"))

    @pytest.mark.parametrize("kind,rate", [
        (PaymentKind.PROFESSIONAL_FEE, "0.06"), (PaymentKind.DIVIDEND, "0.15")])
    def test_withholding_rates(self, kind, rate):
        result = WithholdingTax(kind).calculate(
            TaxContext(period=PERIOD, receipts=Money("10000000")))
        assert result.payable == Money("10000000") * Decimal(rate)

    def test_withholding_is_described_as_a_credit(self):
        result = WithholdingTax().calculate(
            TaxContext(period=PERIOD, receipts=Money("5000000")))
        assert any("credit" in note for note in result.notes)

    def test_the_supply_threshold(self):
        tax = WithholdingTax(PaymentKind.GOODS_AND_SERVICES)
        below = tax.calculate(TaxContext(period=PERIOD, receipts=Money("500000")))
        above = tax.calculate(TaxContext(period=PERIOD, receipts=Money("2000000")))
        assert below.payable.is_none() and above.payable == Money("120000")

    def test_an_unknown_payment_kind_rejected(self):
        with pytest.raises(ValueError):
            WithholdingTax("professional fee")


class TestBusinessIncome:
    def test_it_delegates_to_the_individual_bands(self):
        """A consultant pays the same rates, on a different base."""
        tax = BusinessIncomeTax(expense_allowance=Decimal("0"))
        # 12m of fees with no expenses is 1m a month, same as a 1m salary
        result = tax.calculate(TaxContext(period=PERIOD,
                                          receipts=Money("12000000")))
        salary = PayeTax().calculate(resident(1_000_000)).payable
        assert result.payable == salary * Decimal("12")

    def test_the_allowance_reduces_the_base(self):
        high = BusinessIncomeTax(expense_allowance=Decimal("0.50"))
        low = BusinessIncomeTax(expense_allowance=Decimal("0.10"))
        ctx = TaxContext(period=PERIOD, receipts=Money("60000000"))
        assert high.calculate(ctx).payable < low.calculate(ctx).payable

    def test_actual_expenses_override_the_allowance(self):
        tax = BusinessIncomeTax()
        result = tax.calculate(TaxContext(
            period=PERIOD, receipts=Money("60000000"),
            settings={"expenses": Money("40000000")}))
        assert result.charged_on == Money("20000000")
        assert any("actual expenses" in note for note in result.notes)

    def test_expenses_cannot_exceed_fees(self):
        result = BusinessIncomeTax().calculate(TaxContext(
            period=PERIOD, receipts=Money("1000000"),
            settings={"expenses": Money("5000000")}))
        assert result.charged_on.is_none()

    def test_the_allowance_is_flagged_as_illustrative(self):
        result = BusinessIncomeTax().calculate(
            TaxContext(period=PERIOD, receipts=Money("60000000")))
        assert any("illustrative" in note for note in result.notes)

    @pytest.mark.parametrize("allowance", [Decimal("1.0"), Decimal("-0.1"), 0.3])
    def test_impossible_allowances_rejected(self, allowance):
        with pytest.raises(BadRate):
            BusinessIncomeTax(expense_allowance=allowance)

    @given(fees=st.integers(min_value=0, max_value=500_000_000))
    @settings(max_examples=100, deadline=None)
    def test_business_tax_properties(self, fees):
        result = BusinessIncomeTax().calculate(
            TaxContext(period=PERIOD, receipts=Money(fees)))
        assert not result.payable.is_negative()
        assert result.payable <= Money(fees)


class TestEngagementExtension:
    """Employment against consultancy, at equal cost to the engager."""

    @pytest.fixture
    def planner(self):
        return EngagementPlanner(PERIOD)

    def test_employment_spends_the_whole_outlay(self, planner):
        route = planner.as_employee(Money("11000000"))
        employer = dict(route.charges)["NSSF employer 10%"]
        assert route.received_before_tax + employer == route.outlay

    def test_consultancy_pays_the_whole_outlay_as_fees(self, planner):
        route = planner.as_consultant(Money("5000000"))
        assert route.received_before_tax == route.outlay

    def test_consultancy_attracts_no_nssf(self, planner):
        route = planner.as_consultant(Money("5000000"))
        assert route.retirement_saving.is_none()
        assert not any("NSSF" in label for label, _ in route.charges)

    def test_counting_nssf_narrows_the_gap_substantially(self, planner):
        """The honest comparison: NSSF is the worker's money, not a tax."""
        result = planner.compare(Money("5000000"))
        assert result.value_margin < result.cash_margin
        assert result.nssf_closes == result.employment.retirement_saving

    def test_consultancy_wins_on_cash_even_with_no_expense_allowance(self, planner):
        """The allowance is not what drives the result; NSSF is."""
        strict = EngagementPlanner(
            PERIOD, business=BusinessIncomeTax(expense_allowance=Decimal("0")))
        assert strict.compare(Money("5000000")).better_for_cash == "Consultancy"

    def test_the_sensitivity_table_covers_the_allowances_given(self, planner):
        table = planner.sensitivity(Money("5000000"),
                                    [Decimal("0.1"), Decimal("0.3")])
        assert len(table) == 2
        assert "consultant keeps" in table.columns

    def test_negative_outlay_refused(self, planner):
        with pytest.raises(ValueError):
            planner.as_employee(Money("-1"))
        with pytest.raises(ValueError):
            planner.as_consultant(Money("-1"))

    def test_the_caveats_are_stated_in_the_output(self):
        caveats = Comparison.caveats()
        assert len(caveats) >= 4
        joined = " ".join(caveats).lower()
        assert "question of fact" in joined        # re-characterisation risk
        assert "nssf" in joined                    # the real driver
        assert "vat" in joined                     # registration threshold

    def test_the_sweep_reports_both_measures(self, planner):
        table = planner.sweep([Money("2000000"), Money("10000000")])
        assert "better on cash" in table.columns
        assert "better on total value" in table.columns


class TestTheAbstraction:
    def test_tax_is_abstract(self):
        with pytest.raises(TypeError):
            Tax()

    def test_every_tax_keeps_the_contract(self):
        ctx = TaxContext(period=PERIOD, earnings=Money("1000000"),
                         receipts=Money("5000000"), outgoings=Money("1000000"))
        for tax in (PayeTax(), VatTax(), CompanyTax(), WithholdingTax(),
                    BusinessIncomeTax()):
            assert isinstance(tax, Tax)
            result = tax.calculate(ctx)
            assert isinstance(result, TaxResult)
            assert not result.payable.is_negative()

    def test_a_negative_result_refused(self):
        with pytest.raises(InvalidIncomeError):
            TaxResult("invented", Money("100"), Money("-1"))

    def test_results_show_their_working(self):
        assert "effective" in PayeTax().calculate(resident(5_000_000)).working()

    def test_all_errors_descend_from_tax_error(self):
        for error in (InvalidIncomeError, BadRate, NoScheduleForDateError,
                      UnknownSupply):
            assert issubclass(error, TaxError)
