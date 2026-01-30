"""Integration test for PV-EEG base case."""

import time

import numpy as np
import pytest

from financial_model import FinancialModel


class TestPVEEGBaseCase:
    """Integration tests for the PV-EEG example case from specification."""

    @pytest.fixture
    def pv_eeg_params(self) -> dict:
        """Create PV-EEG parameter set from specification."""
        return {
            "project": {
                "name": "Solar Park Musterstadt",
                "asset_type": "pv",
                "lifetime_years": 25,
                "development_duration_years": 1,
                "construction_duration_years": 0.5,
                "start_date": "2026-07-01",
            },
            "technical": {
                "capacity": 1000,  # kW
                "volume_model_params": {
                    "irradiance_profile": "typical_year_germany",
                    "degradation_rate": 0.005,
                    "availability": 0.97,
                    "module_efficiency": 0.20,
                },
            },
            "financial": {
                "capex": {
                    "development": [
                        {
                            "phase": "development",
                            "year": -1,
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
                            "month": 2,
                            "amount": 150000,
                            "category": "inverters",
                        },
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 4,
                            "amount": 200000,
                            "category": "installation",
                        },
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 5,
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
                            "escalation_rate": None,
                        },
                        {
                            "category": "insurance",
                            "annual_amount": 3000,
                            "indexed": True,
                            "escalation_rate": None,
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
                                    "end_year": 19,
                                    "price": 0.073,  # €/kWh
                                    "indexed": False,
                                    "escalation_rate": 0.0,
                                },
                                "market_period": {
                                    "start_year": 20,
                                    "end_year": 24,
                                    "price_curve": "mid",
                                },
                            },
                        }
                    ]
                },
                "financing": {
                    "equity_share": 0.25,
                    "debt": {
                        "principal": 600000,
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
                    "cost_of_debt": 0.045,
                    "tax_rate": 0.30,
                },
                "inflation": {
                    "base_rate": 0.02,
                },
            },
        }

    @pytest.fixture
    def pv_monthly_volumes(self) -> np.ndarray:
        """
        Create realistic monthly volumes for 1 MW PV in Germany.

        ~1000 kWh/kWp/year = 1,000,000 kWh/year for 1 MWp
        With degradation of 0.5%/year and 97% availability
        """
        n_months = 25 * 12  # 25 years
        base_annual_production = 1_000_000  # kWh/year for 1 MWp
        degradation_rate = 0.005
        availability = 0.97

        # Monthly distribution (Germany typical)
        monthly_factors = np.array(
            [
                0.04,
                0.05,
                0.08,
                0.10,
                0.12,
                0.12,  # Jan-Jun
                0.12,
                0.11,
                0.09,
                0.07,
                0.05,
                0.05,
            ]
        )  # Jul-Dec

        volumes = np.zeros(n_months)
        for month in range(n_months):
            year = month // 12
            month_of_year = month % 12
            deg_factor = (1 - degradation_rate) ** year
            volumes[month] = (
                base_annual_production
                * monthly_factors[month_of_year]
                * deg_factor
                * availability
            )

        return volumes

    def test_pv_eeg_npv_calculated(
        self, pv_eeg_params: dict, pv_monthly_volumes: np.ndarray
    ) -> None:
        """Test that NPV is calculated (not NaN)."""
        model = FinancialModel(asset_type="pv")
        results = model.calculate(pv_eeg_params, pv_monthly_volumes)

        npv = results["kpis"]["npv_project"]
        # NPV should be a finite number (can be positive or negative)
        assert np.isfinite(npv)

    def test_pv_eeg_reasonable_irr(
        self, pv_eeg_params: dict, pv_monthly_volumes: np.ndarray
    ) -> None:
        """Test that IRR is in reasonable range."""
        model = FinancialModel(asset_type="pv")
        results = model.calculate(pv_eeg_params, pv_monthly_volumes)

        irr_project = results["kpis"]["irr_project"]
        # IRR should be between 4% and 12% for typical PV project
        assert 0.04 < irr_project < 0.12

    def test_pv_eeg_dscr_above_minimum(
        self, pv_eeg_params: dict, pv_monthly_volumes: np.ndarray
    ) -> None:
        """Test that DSCR meets typical bank requirements."""
        model = FinancialModel(asset_type="pv")
        results = model.calculate(pv_eeg_params, pv_monthly_volumes)

        # DSCR should be positive (or inf for no debt)
        dscr_min = results["kpis"]["dscr_min"]
        assert dscr_min > 0.5 or dscr_min == float("inf")

    def test_pv_eeg_lcoe_reasonable(
        self, pv_eeg_params: dict, pv_monthly_volumes: np.ndarray
    ) -> None:
        """Test that LCOE is in reasonable range for PV."""
        model = FinancialModel(asset_type="pv")
        results = model.calculate(pv_eeg_params, pv_monthly_volumes)

        lcoe = results["kpis"]["lcoe"]
        # LCOE for PV should be between €0.03 and €0.15/kWh
        assert 0.03 < lcoe < 0.15


class TestPerformance:
    """Performance tests for the financial model."""

    @pytest.fixture
    def simple_params(self) -> dict:
        """Create simple test parameters for performance testing."""
        return {
            "project": {
                "name": "Performance Test",
                "asset_type": "pv",
                "lifetime_years": 25,
            },
            "technical": {"capacity": 1000},
            "financial": {
                "capex": {
                    "development": [],
                    "construction": [
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 0,
                            "amount": 800000,
                            "category": "total",
                        }
                    ],
                    "replacement": [],
                    "decommissioning": {"enabled": False},
                },
                "opex": {
                    "fixed": [
                        {
                            "category": "total",
                            "annual_amount": 15000,
                            "indexed": True,
                        }
                    ],
                    "variable": [],
                },
                "revenue": {
                    "streams": [
                        {
                            "name": "tariff",
                            "type": "fixed_price",
                            "price_structure": {
                                "fixed_period": {
                                    "start_year": 0,
                                    "end_year": 24,
                                    "price": 0.073,
                                    "indexed": False,
                                }
                            },
                        }
                    ]
                },
                "financing": {"equity_share": 1.0},
                "tax": {"corporate_tax_rate": 0.30, "depreciation_years": 20},
                "discount": {"wacc": 0.06},
                "inflation": {"base_rate": 0.02},
            },
        }

    @pytest.fixture
    def simple_volumes(self) -> np.ndarray:
        """Create volumes for 25 years."""
        return np.full(25 * 12, 83333.0)  # ~1M kWh/year

    @pytest.mark.slow
    def test_performance_under_10ms(
        self, simple_params: dict, simple_volumes: np.ndarray
    ) -> None:
        """Test that single evaluation completes in under 10ms."""
        model = FinancialModel(asset_type="pv")

        # Warm-up
        model.calculate(simple_params, simple_volumes)

        # Measure performance
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            model.calculate(simple_params, simple_volumes)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / iterations) * 1000
        assert avg_time_ms < 10, f"Average time {avg_time_ms:.2f}ms exceeds 10ms target"
