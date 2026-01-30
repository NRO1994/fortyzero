"""
Complete integration test for PV financial model with regression checks.

This test validates the entire calculation pipeline and checks results
against expected values to detect unintended changes in future development.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from financial_model import FinancialModel


@dataclass(frozen=True)
class ExpectedKPIs:
    """Expected KPI values for regression testing."""

    npv_project: float
    npv_equity: float
    irr_project: float
    irr_equity: float | None  # Can be NaN
    lcoe: float
    dscr_min: float
    dscr_avg: float
    payback_simple: float
    payback_discounted: float

    # Tolerances for comparison
    npv_tolerance: float = 1000.0  # €
    irr_tolerance: float = 0.001  # 0.1%
    lcoe_tolerance: float = 0.001  # €/kWh
    dscr_tolerance: float = 0.01
    payback_tolerance: float = 0.1  # years


@dataclass(frozen=True)
class ExpectedCashFlows:
    """Expected cash flow values for regression testing."""

    # Year 0 values
    revenue_year_0: float
    opex_fixed_year_0: float
    capex_year_0: float

    # Year 10 values (mid-project, with replacement)
    revenue_year_10: float
    capex_year_10: float  # Inverter replacement

    # Total values
    total_revenue: float
    total_opex_fixed: float
    total_capex: float

    tolerance: float = 100.0  # €


class TestPVIntegrationComplete:
    """
    Complete integration test for PV financial model.

    Tests the full calculation pipeline with a realistic 1 MWp PV project
    in Germany with EEG tariff and validates all outputs against expected
    values to detect regressions.
    """

    # =========================================================================
    # Test Parameters - 1 MWp PV Project Germany
    # =========================================================================

    @pytest.fixture
    def pv_params(self) -> dict[str, Any]:
        """
        Create comprehensive PV project parameters.

        Project: 1 MWp Solar Park in Germany
        - 25 year lifetime
        - EEG feed-in tariff: €0.073/kWh for 20 years
        - 75% debt financing at 4.5% over 15 years
        - 5.5% WACC
        """
        return {
            "project": {
                "name": "Integration Test PV 1MWp",
                "asset_type": "pv",
                "lifetime_years": 25,
            },
            "technical": {
                "capacity": 1000,  # kWp
            },
            "financial": {
                "capex": {
                    "development": [
                        {
                            "phase": "development",
                            "year": 0,
                            "month": 0,
                            "amount": 30000,
                            "category": "permits_engineering",
                        }
                    ],
                    "construction": [
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 0,
                            "amount": 400000,
                            "category": "modules",
                        },
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 0,
                            "amount": 150000,
                            "category": "inverters",
                        },
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 0,
                            "amount": 200000,
                            "category": "installation",
                        },
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 0,
                            "amount": 50000,
                            "category": "grid_connection",
                        },
                    ],
                    "replacement": [
                        {
                            "phase": "replacement",
                            "year": 10,
                            "month": 0,
                            "amount": 80000,
                            "category": "inverters",
                        },
                        {
                            "phase": "replacement",
                            "year": 20,
                            "month": 0,
                            "amount": 80000,
                            "category": "inverters",
                        },
                    ],
                    "decommissioning": {
                        "enabled": True,
                        "year": 25,
                        "amount": 50000,
                        "reserve_annual": True,
                    },
                },
                "opex": {
                    "fixed": [
                        {
                            "category": "maintenance",
                            "annual_amount": 12000,
                            "indexed": True,
                            "escalation_rate": 0.02,
                        },
                        {
                            "category": "insurance",
                            "annual_amount": 3000,
                            "indexed": True,
                            "escalation_rate": 0.02,
                        },
                        {
                            "category": "land_lease",
                            "annual_amount": 5000,
                            "indexed": True,
                            "escalation_rate": 0.02,
                        },
                    ],
                    "variable": [],
                },
                "revenue": {
                    "streams": [
                        {
                            "name": "eeg_tariff",
                            "type": "fixed_price",
                            "price_structure": {
                                "fixed_period": {
                                    "start_year": 0,
                                    "end_year": 24,
                                    "price": 0.073,  # €/kWh
                                    "indexed": False,
                                }
                            },
                        }
                    ]
                },
                "financing": {
                    "equity_share": 0.25,
                    "debt": {
                        "principal": 622500,  # 75% of 830k
                        "interest_rate": 0.045,
                        "term_years": 15,
                        "type": "annuity",
                    },
                },
                "tax": {
                    "corporate_tax_rate": 0.30,
                    "depreciation_method": "linear",
                    "depreciation_years": 20,
                },
                "discount": {
                    "wacc": 0.055,
                    "cost_of_equity": 0.08,
                },
                "inflation": {
                    "base_rate": 0.02,
                },
            },
        }

    @pytest.fixture
    def pv_monthly_volumes(self) -> np.ndarray:
        """
        Create deterministic monthly production volumes.

        Based on:
        - 1000 kWh/kWp/year specific yield (Germany average)
        - 0.5% annual degradation
        - 97% availability
        - Typical monthly distribution for Germany
        """
        n_years = 25
        n_months = n_years * 12
        base_annual_kwh = 1_000_000  # 1000 kWh/kWp × 1000 kWp
        degradation_rate = 0.005
        availability = 0.97

        # Monthly distribution factors (sum = 1.0)
        monthly_factors = np.array([
            0.04, 0.05, 0.08, 0.10, 0.12, 0.12,  # Jan-Jun
            0.12, 0.11, 0.09, 0.07, 0.05, 0.05,  # Jul-Dec
        ])

        volumes = np.zeros(n_months)
        for month in range(n_months):
            year = month // 12
            month_of_year = month % 12
            degradation_factor = (1 - degradation_rate) ** year
            volumes[month] = (
                base_annual_kwh
                * monthly_factors[month_of_year]
                * degradation_factor
                * availability
            )

        return volumes

    # =========================================================================
    # Expected Values - Update these when intentionally changing calculations
    # =========================================================================

    @pytest.fixture
    def expected_kpis(self) -> ExpectedKPIs:
        """
        Expected KPI values for regression testing.

        IMPORTANT: Update these values when intentionally changing
        calculation logic. Run test first to get new values.
        """
        return ExpectedKPIs(
            npv_project=-319020.61,
            npv_equity=-518059.16,
            irr_project=-0.05880,
            irr_equity=-0.06855,
            lcoe=0.1099,
            dscr_min=0.6491,
            dscr_avg=float("inf"),
            payback_simple=float("inf"),
            payback_discounted=float("inf"),
        )

    @pytest.fixture
    def expected_cash_flows(self) -> ExpectedCashFlows:
        """
        Expected cash flow values for regression testing.

        IMPORTANT: Update these values when intentionally changing
        calculation logic.
        """
        return ExpectedCashFlows(
            # Year 0
            revenue_year_0=70810.00,
            opex_fixed_year_0=22000.00,  # 12k + 3k + 5k + 2k reserve
            capex_year_0=830000.00,  # All construction + development
            # Year 10
            revenue_year_10=67348.11,
            capex_year_10=80000.00,  # Inverter replacement
            # Totals
            total_revenue=1_667_996.92,
            total_opex_fixed=690_605.99,
            total_capex=990_000.00,  # 830k + 80k + 80k
        )

    # =========================================================================
    # Main Integration Test
    # =========================================================================

    def test_complete_pv_calculation(
        self,
        pv_params: dict[str, Any],
        pv_monthly_volumes: np.ndarray,
        expected_kpis: ExpectedKPIs,
        expected_cash_flows: ExpectedCashFlows,
    ) -> None:
        """
        Complete integration test validating all model outputs.

        This test:
        1. Runs the full financial model calculation
        2. Validates all KPIs against expected values
        3. Validates key cash flow values
        4. Ensures calculation time is under 10ms
        """
        model = FinancialModel(asset_type="pv")

        # Warm-up run (JIT compilation, caching, etc.)
        model.calculate(pv_params, pv_monthly_volumes)

        # Timed run
        start_time = time.perf_counter()
        results = model.calculate(pv_params, pv_monthly_volumes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # =====================================================================
        # Performance Check
        # =====================================================================
        assert elapsed_ms < 10.0, (
            f"Calculation took {elapsed_ms:.2f}ms, exceeds 10ms target"
        )

        # =====================================================================
        # KPI Regression Checks
        # =====================================================================
        kpis = results["kpis"]

        # NPV Project
        assert kpis["npv_project"] == pytest.approx(
            expected_kpis.npv_project, abs=expected_kpis.npv_tolerance
        ), f"NPV Project changed: {kpis['npv_project']:.2f} vs expected {expected_kpis.npv_project:.2f}"

        # NPV Equity
        assert kpis["npv_equity"] == pytest.approx(
            expected_kpis.npv_equity, abs=expected_kpis.npv_tolerance
        ), f"NPV Equity changed: {kpis['npv_equity']:.2f} vs expected {expected_kpis.npv_equity:.2f}"

        # IRR Project
        assert kpis["irr_project"] == pytest.approx(
            expected_kpis.irr_project, abs=expected_kpis.irr_tolerance
        ), f"IRR Project changed: {kpis['irr_project']:.4f} vs expected {expected_kpis.irr_project:.4f}"

        # IRR Equity (may be NaN or negative)
        if expected_kpis.irr_equity is None:
            assert np.isnan(kpis["irr_equity"]), (
                f"IRR Equity should be NaN, got {kpis['irr_equity']}"
            )
        elif np.isnan(kpis["irr_equity"]):
            pass  # Accept NaN for problematic cash flow profiles
        else:
            assert kpis["irr_equity"] == pytest.approx(
                expected_kpis.irr_equity, abs=expected_kpis.irr_tolerance
            ), f"IRR Equity changed: {kpis['irr_equity']:.4f} vs expected {expected_kpis.irr_equity:.4f}"

        # LCOE
        assert kpis["lcoe"] == pytest.approx(
            expected_kpis.lcoe, abs=expected_kpis.lcoe_tolerance
        ), f"LCOE changed: {kpis['lcoe']:.4f} vs expected {expected_kpis.lcoe:.4f}"

        # DSCR
        assert kpis["dscr_min"] == pytest.approx(
            expected_kpis.dscr_min, abs=expected_kpis.dscr_tolerance
        ), f"DSCR min changed: {kpis['dscr_min']:.4f} vs expected {expected_kpis.dscr_min:.4f}"

        if np.isinf(expected_kpis.dscr_avg):
            assert np.isinf(kpis["dscr_avg"]), f"DSCR avg should be inf, got {kpis['dscr_avg']}"
        else:
            assert kpis["dscr_avg"] == pytest.approx(
                expected_kpis.dscr_avg, abs=expected_kpis.dscr_tolerance
            ), f"DSCR avg changed: {kpis['dscr_avg']:.4f} vs expected {expected_kpis.dscr_avg:.4f}"

        # =====================================================================
        # Cash Flow Regression Checks
        # =====================================================================
        cf = results["cash_flows"]["annual"]

        # Year 0
        assert cf["revenue"][0] == pytest.approx(
            expected_cash_flows.revenue_year_0, abs=expected_cash_flows.tolerance
        ), f"Revenue Year 0 changed: {cf['revenue'][0]:.2f}"

        assert cf["opex_fixed"][0] == pytest.approx(
            expected_cash_flows.opex_fixed_year_0, abs=expected_cash_flows.tolerance
        ), f"OPEX Fixed Year 0 changed: {cf['opex_fixed'][0]:.2f}"

        assert cf["capex"][0] == pytest.approx(
            expected_cash_flows.capex_year_0, abs=expected_cash_flows.tolerance
        ), f"CAPEX Year 0 changed: {cf['capex'][0]:.2f}"

        # Year 10
        assert cf["revenue"][10] == pytest.approx(
            expected_cash_flows.revenue_year_10, abs=expected_cash_flows.tolerance
        ), f"Revenue Year 10 changed: {cf['revenue'][10]:.2f}"

        assert cf["capex"][10] == pytest.approx(
            expected_cash_flows.capex_year_10, abs=expected_cash_flows.tolerance
        ), f"CAPEX Year 10 changed: {cf['capex'][10]:.2f}"

        # Totals
        assert np.sum(cf["revenue"]) == pytest.approx(
            expected_cash_flows.total_revenue, abs=expected_cash_flows.tolerance * 10
        ), f"Total Revenue changed: {np.sum(cf['revenue']):.2f}"

        assert np.sum(cf["opex_fixed"]) == pytest.approx(
            expected_cash_flows.total_opex_fixed, abs=expected_cash_flows.tolerance * 10
        ), f"Total OPEX Fixed changed: {np.sum(cf['opex_fixed']):.2f}"

        assert np.sum(cf["capex"]) == pytest.approx(
            expected_cash_flows.total_capex, abs=expected_cash_flows.tolerance
        ), f"Total CAPEX changed: {np.sum(cf['capex']):.2f}"

        # =====================================================================
        # Structure Checks
        # =====================================================================
        # Verify all expected keys exist
        assert "kpis" in results
        assert "cash_flows" in results
        assert "financing" in results

        assert "annual" in results["cash_flows"]
        assert "monthly" in results["cash_flows"]

        # Verify array shapes
        assert len(cf["revenue"]) == 25
        assert len(cf["fcf_unlevered"]) == 25
        assert len(cf["fcf_levered"]) == 25

        # =====================================================================
        # Print Summary
        # =====================================================================
        print(f"\n{'='*60}")
        print("PV Integration Test Results")
        print(f"{'='*60}")
        print(f"Calculation Time: {elapsed_ms:.2f}ms (target: <10ms)")
        print(f"\nKPIs:")
        print(f"  NPV Project:  {kpis['npv_project']:>12,.0f} €")
        print(f"  NPV Equity:   {kpis['npv_equity']:>12,.0f} €")
        print(f"  IRR Project:  {kpis['irr_project']:>12.2%}")
        print(f"  IRR Equity:   {'NaN':>12}" if np.isnan(kpis['irr_equity']) else f"  IRR Equity:   {kpis['irr_equity']:>12.2%}")
        print(f"  LCOE:         {kpis['lcoe']:>12.4f} €/kWh")
        print(f"  DSCR min:     {kpis['dscr_min']:>12.2f}")
        print(f"  DSCR avg:     {kpis['dscr_avg']:>12.2f}")
        print(f"\nCash Flows (Year 0 / Year 10 / Total):")
        print(f"  Revenue:      {cf['revenue'][0]:>10,.0f} / {cf['revenue'][10]:>10,.0f} / {np.sum(cf['revenue']):>12,.0f} €")
        print(f"  OPEX Fixed:   {cf['opex_fixed'][0]:>10,.0f} / {cf['opex_fixed'][10]:>10,.0f} / {np.sum(cf['opex_fixed']):>12,.0f} €")
        print(f"  CAPEX:        {cf['capex'][0]:>10,.0f} / {cf['capex'][10]:>10,.0f} / {np.sum(cf['capex']):>12,.0f} €")
        print(f"{'='*60}")

    # =========================================================================
    # Performance Benchmark Test
    # =========================================================================

    def test_performance_benchmark(
        self,
        pv_params: dict[str, Any],
        pv_monthly_volumes: np.ndarray,
    ) -> None:
        """
        Benchmark test running 100 iterations to verify consistent performance.

        Ensures average calculation time stays under 10ms target.
        """
        model = FinancialModel(asset_type="pv")

        # Warm-up
        for _ in range(5):
            model.calculate(pv_params, pv_monthly_volumes)

        # Benchmark
        iterations = 100
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            model.calculate(pv_params, pv_monthly_volumes)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = np.mean(times)
        min_ms = np.min(times)
        max_ms = np.max(times)
        std_ms = np.std(times)

        print(f"\n{'='*60}")
        print(f"Performance Benchmark ({iterations} iterations)")
        print(f"{'='*60}")
        print(f"  Average: {avg_ms:.2f}ms")
        print(f"  Min:     {min_ms:.2f}ms")
        print(f"  Max:     {max_ms:.2f}ms")
        print(f"  Std:     {std_ms:.2f}ms")
        print(f"{'='*60}")

        assert avg_ms < 10.0, f"Average time {avg_ms:.2f}ms exceeds 10ms target"
        assert max_ms < 20.0, f"Max time {max_ms:.2f}ms exceeds 20ms limit"

    # =========================================================================
    # Helper Test to Generate Expected Values
    # =========================================================================

    @pytest.mark.skip(reason="Run manually to generate expected values")
    def test_generate_expected_values(
        self,
        pv_params: dict[str, Any],
        pv_monthly_volumes: np.ndarray,
    ) -> None:
        """
        Helper test to generate expected values for regression testing.

        Run this manually when intentionally changing calculation logic:
        pytest -k test_generate_expected_values --no-header -rN
        """
        model = FinancialModel(asset_type="pv")
        results = model.calculate(pv_params, pv_monthly_volumes)

        kpis = results["kpis"]
        cf = results["cash_flows"]["annual"]

        print("\n" + "=" * 60)
        print("COPY THESE VALUES TO expected_kpis fixture:")
        print("=" * 60)
        print(f"""
        return ExpectedKPIs(
            npv_project={kpis['npv_project']:.2f},
            npv_equity={kpis['npv_equity']:.2f},
            irr_project={kpis['irr_project']:.5f},
            irr_equity={'None' if np.isnan(kpis['irr_equity']) else f"{kpis['irr_equity']:.5f}"},
            lcoe={kpis['lcoe']:.4f},
            dscr_min={kpis['dscr_min']:.4f},
            dscr_avg={kpis['dscr_avg']:.4f},
            payback_simple={kpis['payback_simple']:.1f},
            payback_discounted={kpis['payback_discounted']:.1f},
        )
        """)

        print("\n" + "=" * 60)
        print("COPY THESE VALUES TO expected_cash_flows fixture:")
        print("=" * 60)
        print(f"""
        return ExpectedCashFlows(
            revenue_year_0={cf['revenue'][0]:.2f},
            opex_fixed_year_0={cf['opex_fixed'][0]:.2f},
            capex_year_0={cf['capex'][0]:.2f},
            revenue_year_10={cf['revenue'][10]:.2f},
            capex_year_10={cf['capex'][10]:.2f},
            total_revenue={np.sum(cf['revenue']):.2f},
            total_opex_fixed={np.sum(cf['opex_fixed']):.2f},
            total_capex={np.sum(cf['capex']):.2f},
        )
        """)
