"""Tests for hourly revenue calculations with CSV price curves."""

from pathlib import Path

import numpy as np
import pytest

from financial_model import FinancialModel
from financial_model.models.csv_price_curve import CSVPriceCurve
from financial_model.models.revenue_model import RevenueModel


# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_PRICES_2Y = TEST_DATA_DIR / "sample_prices_2y.csv"


class TestCSVPriceCurve:
    """Tests for CSVPriceCurve class."""

    @pytest.fixture
    def price_curve_mid(self) -> CSVPriceCurve:
        """Create mid scenario price curve."""
        return CSVPriceCurve(scenario="mid", csv_path=SAMPLE_PRICES_2Y)

    def test_init_loads_data(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test that initialization loads price data."""
        assert price_curve_mid._prices is not None
        assert len(price_curve_mid._prices) > 0

    def test_available_years(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test available_years property."""
        assert price_curve_mid.available_years == 2

    def test_get_hourly_prices_shape(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test that get_hourly_prices returns correct shape."""
        prices = price_curve_mid.get_hourly_prices(start_year=0, end_year=0)
        assert len(prices) == 8760  # 1 year

        prices = price_curve_mid.get_hourly_prices(start_year=0, end_year=1)
        assert len(prices) == 17520  # 2 years

    def test_get_hourly_prices_positive(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test that all prices are positive."""
        prices = price_curve_mid.get_hourly_prices(start_year=0, end_year=1)
        assert np.all(prices > 0)

    def test_scenarios_differ(self) -> None:
        """Test that different scenarios have different prices."""
        high = CSVPriceCurve(scenario="high", csv_path=SAMPLE_PRICES_2Y)
        mid = CSVPriceCurve(scenario="mid", csv_path=SAMPLE_PRICES_2Y)
        low = CSVPriceCurve(scenario="low", csv_path=SAMPLE_PRICES_2Y)

        high_prices = high.get_hourly_prices(0, 0)
        mid_prices = mid.get_hourly_prices(0, 0)
        low_prices = low.get_hourly_prices(0, 0)

        assert np.mean(high_prices) > np.mean(mid_prices)
        assert np.mean(mid_prices) > np.mean(low_prices)

    def test_extrapolation_beyond_data(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test that extrapolation works for years beyond available data."""
        # Request year 5 when only 2 years available
        prices = price_curve_mid.get_hourly_prices(start_year=5, end_year=5)

        assert len(prices) == 8760
        assert np.all(prices > 0)

        # Extrapolated prices should be higher due to 2% escalation
        base_prices = price_curve_mid.get_hourly_prices(0, 0)
        # Year 5 vs Year 1 (last available) = 4 years of escalation
        expected_factor = (1.02) ** 4
        assert np.mean(prices) > np.mean(base_prices)

    def test_invalid_scenario_raises(self) -> None:
        """Test that invalid scenario raises error."""
        with pytest.raises(ValueError, match="Invalid scenario"):
            CSVPriceCurve(scenario="invalid", csv_path=SAMPLE_PRICES_2Y)

    def test_missing_file_raises(self) -> None:
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            CSVPriceCurve(scenario="mid", csv_path="nonexistent.csv")

    def test_get_monthly_prices(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test monthly price aggregation."""
        monthly = price_curve_mid.get_monthly_prices(start_year=0, end_year=0)
        assert len(monthly) == 12

    def test_price_statistics(self, price_curve_mid: CSVPriceCurve) -> None:
        """Test price statistics method."""
        stats = price_curve_mid.get_price_statistics()

        assert stats["scenario"] == "mid"
        assert stats["years"] == 2
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert stats["max"] > stats["min"]


class TestRevenueModelHourly:
    """Tests for hourly revenue calculations in RevenueModel."""

    @pytest.fixture
    def revenue_model(self) -> RevenueModel:
        """Create RevenueModel instance."""
        return RevenueModel()

    @pytest.fixture
    def hourly_volumes(self) -> np.ndarray:
        """Create test hourly volumes for 2 years."""
        # Constant 100 kWh per hour
        return np.full(2 * 8760, 100.0)

    @pytest.fixture
    def price_curve(self) -> CSVPriceCurve:
        """Create price curve for testing."""
        return CSVPriceCurve(scenario="mid", csv_path=SAMPLE_PRICES_2Y)

    def test_fixed_price_hourly(
        self, revenue_model: RevenueModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test hourly revenue with fixed price."""
        n_years = 2
        config = {
            "streams": [
                {
                    "name": "fixed_tariff",
                    "price_structure": {
                        "fixed_period": {
                            "start_year": 0,
                            "end_year": 1,
                            "price": 0.10,  # €/kWh
                            "indexed": False,
                        }
                    },
                }
            ]
        }

        revenue = revenue_model.calculate_hourly(
            hourly_volumes=hourly_volumes,
            revenue_config=config,
            inflation_rates=np.full(n_years, 0.02),
            price_curve=None,
            n_years=n_years,
        )

        # 100 kWh × €0.10/kWh = €10/hour
        expected_hourly = 10.0
        np.testing.assert_array_almost_equal(revenue, np.full(len(revenue), expected_hourly))

    def test_market_price_hourly(
        self,
        revenue_model: RevenueModel,
        hourly_volumes: np.ndarray,
        price_curve: CSVPriceCurve,
    ) -> None:
        """Test hourly revenue with market prices."""
        n_years = 2
        config = {
            "streams": [
                {
                    "name": "merchant",
                    "price_structure": {
                        "market_period": {
                            "start_year": 0,
                            "end_year": 1,
                            "price_curve": "mid",
                        }
                    },
                }
            ]
        }

        revenue = revenue_model.calculate_hourly(
            hourly_volumes=hourly_volumes,
            revenue_config=config,
            inflation_rates=np.full(n_years, 0.02),
            price_curve=price_curve,
            n_years=n_years,
        )

        # Revenue should vary with market prices
        assert np.std(revenue) > 0  # Not all same value
        assert np.all(revenue >= 0)

    def test_mixed_periods_hourly(
        self,
        revenue_model: RevenueModel,
        price_curve: CSVPriceCurve,
    ) -> None:
        """Test transition from fixed to market pricing."""
        # 5 years: first 2 fixed, then 3 market
        n_years = 5
        hourly_volumes = np.full(n_years * 8760, 100.0)

        config = {
            "streams": [
                {
                    "name": "hybrid",
                    "price_structure": {
                        "fixed_period": {
                            "start_year": 0,
                            "end_year": 1,
                            "price": 0.073,
                            "indexed": False,
                        },
                        "market_period": {
                            "start_year": 2,
                            "end_year": 4,
                            "price_curve": "mid",
                        },
                    },
                }
            ]
        }

        revenue = revenue_model.calculate_hourly(
            hourly_volumes=hourly_volumes,
            revenue_config=config,
            inflation_rates=np.full(n_years, 0.02),
            price_curve=price_curve,
            n_years=n_years,
        )

        # Fixed period revenue (year 0-1)
        fixed_revenue = revenue[: 2 * 8760]
        assert np.allclose(fixed_revenue, 100 * 0.073)  # All same

        # Market period revenue (year 2-4)
        market_revenue = revenue[2 * 8760 :]
        assert np.std(market_revenue) > 0  # Varies

    def test_aggregation_hourly_to_annual(
        self, revenue_model: RevenueModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test aggregation from hourly to annual."""
        n_years = 2
        config = {
            "streams": [
                {
                    "name": "fixed",
                    "price_structure": {
                        "fixed_period": {
                            "start_year": 0,
                            "end_year": 1,
                            "price": 0.10,
                            "indexed": False,
                        }
                    },
                }
            ]
        }

        hourly_revenue = revenue_model.calculate_hourly(
            hourly_volumes=hourly_volumes,
            revenue_config=config,
            inflation_rates=np.full(n_years, 0.02),
            price_curve=None,
            n_years=n_years,
        )

        annual = RevenueModel.aggregate_hourly_to_annual(hourly_revenue)

        assert len(annual) == 2
        # 100 kWh × €0.10 × 8760 hours = €87,600/year
        expected_annual = 100 * 0.10 * 8760
        np.testing.assert_array_almost_equal(annual, [expected_annual, expected_annual])


class TestFinancialModelHourly:
    """Tests for FinancialModel with hourly calculations."""

    @pytest.fixture
    def simple_params(self) -> dict:
        """Create simple test parameters."""
        return {
            "project": {
                "name": "Test Project",
                "asset_type": "pv",
                "lifetime_years": 2,
            },
            "technical": {
                "capacity": 1000,  # kW
            },
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
                            "category": "maintenance",
                            "annual_amount": 10000,
                            "indexed": False,
                        }
                    ],
                    "variable": [],
                },
                "revenue": {
                    "streams": [
                        {
                            "name": "merchant",
                            "price_structure": {
                                "market_period": {
                                    "start_year": 0,
                                    "end_year": 1,
                                    "price_curve": "mid",
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
    def hourly_volumes(self) -> np.ndarray:
        """Create hourly volumes for 2 years (simulating PV production)."""
        n_hours = 2 * 8760
        volumes = np.zeros(n_hours)

        for hour in range(n_hours):
            hour_of_day = hour % 24
            # Simple solar pattern
            if 6 <= hour_of_day <= 18:
                volumes[hour] = 500 * np.sin(np.pi * (hour_of_day - 6) / 12)

        return volumes

    @pytest.fixture
    def price_curve(self) -> CSVPriceCurve:
        """Create price curve."""
        return CSVPriceCurve(scenario="mid", csv_path=SAMPLE_PRICES_2Y)

    def test_calculate_hourly_returns_structure(
        self,
        simple_params: dict,
        hourly_volumes: np.ndarray,
        price_curve: CSVPriceCurve,
    ) -> None:
        """Test that calculate_hourly returns expected structure."""
        model = FinancialModel(asset_type="pv", price_curve=price_curve)
        results = model.calculate_hourly(simple_params, hourly_volumes)

        assert "kpis" in results
        assert "cash_flows" in results
        assert "financing" in results

        # Check hourly data is included
        assert "hourly" in results["cash_flows"]
        assert "revenue" in results["cash_flows"]["hourly"]
        assert "volume" in results["cash_flows"]["hourly"]

    def test_hourly_revenue_varies_with_price(
        self,
        simple_params: dict,
        hourly_volumes: np.ndarray,
        price_curve: CSVPriceCurve,
    ) -> None:
        """Test that hourly revenue varies with market prices."""
        model = FinancialModel(asset_type="pv", price_curve=price_curve)
        results = model.calculate_hourly(simple_params, hourly_volumes)

        hourly_revenue = results["cash_flows"]["hourly"]["revenue"]

        # Revenue should have variation (not constant)
        # Only check non-zero values (no production at night)
        nonzero_revenue = hourly_revenue[hourly_revenue > 0]
        assert np.std(nonzero_revenue) > 0

    def test_scenario_affects_revenue(
        self,
        simple_params: dict,
        hourly_volumes: np.ndarray,
    ) -> None:
        """Test that different scenarios produce different revenues."""
        results = {}

        for scenario in ["high", "mid", "low"]:
            price_curve = CSVPriceCurve(scenario=scenario, csv_path=SAMPLE_PRICES_2Y)
            model = FinancialModel(asset_type="pv", price_curve=price_curve)
            results[scenario] = model.calculate_hourly(simple_params, hourly_volumes)

        # Check annual revenues
        high_annual = np.sum(results["high"]["cash_flows"]["annual"]["revenue"])
        mid_annual = np.sum(results["mid"]["cash_flows"]["annual"]["revenue"])
        low_annual = np.sum(results["low"]["cash_flows"]["annual"]["revenue"])

        assert high_annual > mid_annual
        assert mid_annual > low_annual

    def test_kpis_calculated(
        self,
        simple_params: dict,
        hourly_volumes: np.ndarray,
        price_curve: CSVPriceCurve,
    ) -> None:
        """Test that all KPIs are calculated."""
        model = FinancialModel(asset_type="pv", price_curve=price_curve)
        results = model.calculate_hourly(simple_params, hourly_volumes)

        kpis = results["kpis"]
        assert "npv_project" in kpis
        assert "irr_project" in kpis
        assert "lcoe" in kpis
        assert "dscr_min" in kpis


@pytest.mark.slow
class TestHourlyPerformance:
    """Performance tests for hourly calculations."""

    def test_25_year_calculation_time(self, tmp_path: Path) -> None:
        """Test performance of 25-year hourly calculation."""
        import time

        # Generate larger price file
        from financial_model.data.generate_sample_prices import generate_hourly_prices

        df = generate_hourly_prices(n_years=25)
        price_file = tmp_path / "prices_25y.csv"
        df.to_csv(price_file, index=False)

        price_curve = CSVPriceCurve(scenario="mid", csv_path=price_file)

        # Create 25-year volumes
        n_hours = 25 * 8760
        hourly_volumes = np.random.rand(n_hours) * 500

        params = {
            "project": {"name": "Perf Test", "asset_type": "pv", "lifetime_years": 25},
            "technical": {"capacity": 1000},
            "financial": {
                "capex": {
                    "development": [],
                    "construction": [
                        {"phase": "construction", "year": 0, "month": 0, "amount": 800000, "category": "total"}
                    ],
                    "replacement": [],
                    "decommissioning": {"enabled": False},
                },
                "opex": {"fixed": [{"category": "maint", "annual_amount": 10000, "indexed": False}], "variable": []},
                "revenue": {
                    "streams": [
                        {
                            "name": "merchant",
                            "price_structure": {"market_period": {"start_year": 0, "end_year": 24, "price_curve": "mid"}},
                        }
                    ]
                },
                "financing": {"equity_share": 1.0},
                "tax": {"corporate_tax_rate": 0.30, "depreciation_years": 20},
                "discount": {"wacc": 0.06},
                "inflation": {"base_rate": 0.02},
            },
        }

        model = FinancialModel(asset_type="pv", price_curve=price_curve)

        # Warm up
        model.calculate_hourly(params, hourly_volumes)

        # Measure
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            model.calculate_hourly(params, hourly_volumes)
        elapsed = time.time() - start

        avg_ms = (elapsed / iterations) * 1000
        print(f"Average calculation time: {avg_ms:.2f}ms")

        # Should complete reasonably fast (allow more time for hourly)
        assert avg_ms < 500  # 500ms max for 25-year hourly calculation
