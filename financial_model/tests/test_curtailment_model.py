"""Tests for the CurtailmentModel."""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from financial_model.models.curtailment_model import CurtailmentModel


class TestCurtailmentModel:
    """Tests for CurtailmentModel calculations."""

    HOURS_PER_YEAR = 8760

    @pytest.fixture
    def model(self) -> CurtailmentModel:
        """Create CurtailmentModel instance."""
        return CurtailmentModel()

    @pytest.fixture
    def hourly_volumes(self) -> np.ndarray:
        """Create 25 years of hourly production (100 kWh/h constant)."""
        return np.ones(25 * self.HOURS_PER_YEAR) * 100

    @pytest.fixture
    def monthly_volumes(self) -> np.ndarray:
        """Create 25 years of monthly production (100,000 kWh/month constant)."""
        return np.ones(25 * 12) * 100_000


class TestAnnualRatesMode(TestCurtailmentModel):
    """Tests for annual_rates curtailment mode."""

    def test_single_rate_hourly(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test with single curtailment rate on hourly data."""
        params = {"mode": "annual_rates", "rates": 0.05}
        result = model.apply(hourly_volumes, params, n_years=25)

        assert len(result) == len(hourly_volumes)
        np.testing.assert_allclose(result, hourly_volumes * 0.95)

    def test_single_rate_monthly(
        self, model: CurtailmentModel, monthly_volumes: np.ndarray
    ) -> None:
        """Test with single curtailment rate on monthly data."""
        params = {"mode": "annual_rates", "rates": 0.03}
        result = model.apply(monthly_volumes, params, n_years=25)

        assert len(result) == len(monthly_volumes)
        np.testing.assert_allclose(result, monthly_volumes * 0.97)

    def test_array_rates(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test with annually varying curtailment rates."""
        # Rates increase from 0% to 24% over 25 years
        rates = [0.01 * i for i in range(25)]
        params = {"mode": "annual_rates", "rates": rates}
        result = model.apply(hourly_volumes, params, n_years=25)

        # Check each year has correct rate applied
        for year in range(25):
            year_slice = slice(year * self.HOURS_PER_YEAR, (year + 1) * self.HOURS_PER_YEAR)
            expected_factor = 1 - 0.01 * year
            np.testing.assert_allclose(
                result[year_slice],
                hourly_volumes[year_slice] * expected_factor,
                rtol=1e-10,
            )

    def test_rate_extension(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that short rate arrays are extended with last value."""
        # Only provide 5 rates for 25-year project
        rates = [0.01, 0.02, 0.03, 0.04, 0.05]
        params = {"mode": "annual_rates", "rates": rates}
        result = model.apply(hourly_volumes, params, n_years=25)

        # Year 0 should have 1% curtailment
        year_0_slice = slice(0, self.HOURS_PER_YEAR)
        np.testing.assert_allclose(result[year_0_slice], 99.0)

        # Year 24 should have 5% curtailment (last rate repeated)
        year_24_slice = slice(24 * self.HOURS_PER_YEAR, 25 * self.HOURS_PER_YEAR)
        np.testing.assert_allclose(result[year_24_slice], 95.0)

    def test_zero_curtailment(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test with zero curtailment rate."""
        params = {"mode": "annual_rates", "rates": 0.0}
        result = model.apply(hourly_volumes, params, n_years=25)

        np.testing.assert_array_equal(result, hourly_volumes)


class TestTimeseriesMode(TestCurtailmentModel):
    """Tests for timeseries curtailment mode."""

    def test_csv_loading(self, model: CurtailmentModel) -> None:
        """Test loading curtailment factors from CSV."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("hour,curtailment_factor\n")
            for i in range(100):
                factor = 0.9 if i % 10 == 0 else 1.0
                f.write(f"{i},{factor}\n")
            csv_path = f.name

        try:
            volumes = np.ones(100) * 100
            params = {"mode": "timeseries", "csv_path": csv_path}
            result = model.apply(volumes, params, n_years=1)

            # Every 10th hour should have 10% curtailment
            assert result[0] == pytest.approx(90.0)
            assert result[1] == pytest.approx(100.0)
            assert result[10] == pytest.approx(90.0)
        finally:
            Path(csv_path).unlink()

    def test_csv_extension(self, model: CurtailmentModel) -> None:
        """Test that short CSV profile is extended by repeating."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("hour,curtailment_factor\n")
            f.write("0,0.8\n")
            f.write("1,1.0\n")
            csv_path = f.name

        try:
            volumes = np.ones(10) * 100
            params = {"mode": "timeseries", "csv_path": csv_path}
            result = model.apply(volumes, params, n_years=1)

            # Pattern should repeat: 0.8, 1.0, 0.8, 1.0, ...
            expected = np.array([80, 100, 80, 100, 80, 100, 80, 100, 80, 100])
            np.testing.assert_allclose(result, expected)
        finally:
            Path(csv_path).unlink()

    def test_missing_csv_path(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that missing csv_path raises error."""
        params = {"mode": "timeseries"}

        with pytest.raises(ValueError, match="csv_path required"):
            model.apply(hourly_volumes, params, n_years=25)


class TestPriceBasedMode(TestCurtailmentModel):
    """Tests for price_based curtailment mode (§51 EEG)."""

    def test_simple_negative_price_curtailment(
        self, model: CurtailmentModel
    ) -> None:
        """Test basic curtailment at negative prices."""
        volumes = np.ones(10) * 100
        prices = np.array([0.05, 0.03, -0.01, -0.02, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04])

        params = {
            "mode": "price_based",
            "price_threshold": 0.0,
            "curtailment_factor": 1.0,
            "consecutive_hours": 1,  # Disable 3-hour rule for this test
        }

        result = model.apply(volumes, params, n_years=1, hourly_prices=prices)

        # Hours 2-5 have negative prices (below threshold)
        expected = np.array([100, 100, 0, 0, 0, 0, 100, 100, 100, 100])
        np.testing.assert_array_equal(result, expected)

    def test_3_hour_rule(self, model: CurtailmentModel) -> None:
        """Test §51 EEG 3-hour consecutive rule."""
        # 10 hours of production
        volumes = np.ones(10) * 100

        # Prices: only hours 3-6 have 4 consecutive negative hours
        prices = np.array([0.05, -0.01, 0.01, -0.01, -0.02, -0.03, -0.01, 0.02, 0.03, 0.04])
        #                   0     1     2     3      4      5      6     7     8     9
        # Negative:                X           X      X      X      X

        params = {
            "mode": "price_based",
            "price_threshold": 0.0,
            "curtailment_factor": 1.0,
            "consecutive_hours": 3,  # 3-hour rule
        }

        result = model.apply(volumes, params, n_years=1, hourly_prices=prices)

        # Hours 3-6 have 4 consecutive negative, so curtailment applies
        # Hour 1 is isolated negative, so no curtailment
        expected = np.array([100, 100, 100, 0, 0, 0, 0, 100, 100, 100])
        np.testing.assert_array_equal(result, expected)

    def test_3_hour_rule_exactly_3(self, model: CurtailmentModel) -> None:
        """Test that exactly 3 consecutive hours triggers curtailment."""
        volumes = np.ones(10) * 100
        # Exactly 3 consecutive negative hours (indices 2, 3, 4)
        prices = np.array([0.05, 0.03, -0.01, -0.02, -0.03, 0.01, 0.02, 0.03, 0.04, 0.05])

        params = {
            "mode": "price_based",
            "price_threshold": 0.0,
            "curtailment_factor": 1.0,
            "consecutive_hours": 3,
        }

        result = model.apply(volumes, params, n_years=1, hourly_prices=prices)

        # Hours 2-4 should be curtailed (3 consecutive)
        expected = np.array([100, 100, 0, 0, 0, 100, 100, 100, 100, 100])
        np.testing.assert_array_equal(result, expected)

    def test_3_hour_rule_only_2_consecutive(self, model: CurtailmentModel) -> None:
        """Test that only 2 consecutive hours does NOT trigger curtailment."""
        volumes = np.ones(10) * 100
        # Only 2 consecutive negative hours (indices 2, 3)
        prices = np.array([0.05, 0.03, -0.01, -0.02, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05])

        params = {
            "mode": "price_based",
            "price_threshold": 0.0,
            "curtailment_factor": 1.0,
            "consecutive_hours": 3,
        }

        result = model.apply(volumes, params, n_years=1, hourly_prices=prices)

        # No curtailment - only 2 consecutive hours below threshold
        np.testing.assert_array_equal(result, volumes)

    def test_partial_curtailment(self, model: CurtailmentModel) -> None:
        """Test partial curtailment factor."""
        volumes = np.ones(10) * 100
        prices = np.array([-0.01] * 5 + [0.05] * 5)

        params = {
            "mode": "price_based",
            "price_threshold": 0.0,
            "curtailment_factor": 0.5,  # 50% curtailment
            "consecutive_hours": 3,
        }

        result = model.apply(volumes, params, n_years=1, hourly_prices=prices)

        # First 5 hours: 50% curtailment, remaining: no curtailment
        expected = np.array([50, 50, 50, 50, 50, 100, 100, 100, 100, 100])
        np.testing.assert_array_equal(result, expected)

    def test_missing_prices(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that missing hourly_prices raises error."""
        params = {"mode": "price_based", "price_threshold": 0.0}

        with pytest.raises(ValueError, match="hourly_prices required"):
            model.apply(hourly_volumes, params, n_years=25)


class TestCapacityLimitMode(TestCurtailmentModel):
    """Tests for capacity_limit curtailment mode (70% rule)."""

    def test_70_percent_rule(self, model: CurtailmentModel) -> None:
        """Test capacity limitation at 70%."""
        # Production varies between 50 and 150 kWh/h
        volumes = np.array([50, 70, 100, 130, 150, 70, 80, 60, 140, 120])

        params = {"mode": "capacity_limit", "limit_factor": 0.70}
        result = model.apply(volumes, params, n_years=1, capacity_kw=100)

        # Max output: 100 kW × 0.70 = 70 kWh/h
        expected = np.array([50, 70, 70, 70, 70, 70, 70, 60, 70, 70])
        np.testing.assert_array_equal(result, expected)

    def test_different_limit_factors(self, model: CurtailmentModel) -> None:
        """Test with different limit factors."""
        volumes = np.ones(10) * 100

        # 60% limit
        params = {"mode": "capacity_limit", "limit_factor": 0.60}
        result = model.apply(volumes, params, n_years=1, capacity_kw=100)
        np.testing.assert_allclose(result, 60.0)

        # 80% limit
        params = {"mode": "capacity_limit", "limit_factor": 0.80}
        result = model.apply(volumes, params, n_years=1, capacity_kw=100)
        np.testing.assert_allclose(result, 80.0)

    def test_missing_capacity(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that missing capacity_kw raises error."""
        params = {"mode": "capacity_limit", "limit_factor": 0.70}

        with pytest.raises(ValueError, match="capacity_kw required"):
            model.apply(hourly_volumes, params, n_years=25)


class TestStochasticMode(TestCurtailmentModel):
    """Tests for stochastic curtailment mode."""

    def test_reproducibility_with_seed(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that results are reproducible with same seed."""
        params = {
            "mode": "stochastic",
            "base_rate": 0.03,
            "volatility": 0.01,
            "seed": 42,
        }

        result1 = model.apply(hourly_volumes, params, n_years=25)
        result2 = model.apply(hourly_volumes, params, n_years=25)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_different_results(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that different seeds produce different results."""
        params = {
            "mode": "stochastic",
            "base_rate": 0.03,
            "volatility": 0.01,
            "seed": 42,
        }

        result1 = model.apply(hourly_volumes, params, n_years=25)

        params["seed"] = 123
        result2 = model.apply(hourly_volumes, params, n_years=25)

        # Results should differ
        assert not np.array_equal(result1, result2)

    def test_trend_increases_curtailment(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that positive trend increases curtailment over time."""
        params = {
            "mode": "stochastic",
            "base_rate": 0.02,
            "volatility": 0.0,  # No random variation
            "trend": 0.01,  # 1% increase per year
            "seed": 42,
        }

        result = model.apply(hourly_volumes, params, n_years=25)

        # Calculate actual annual production
        original_annual = np.array([
            np.sum(hourly_volumes[y * self.HOURS_PER_YEAR : (y + 1) * self.HOURS_PER_YEAR])
            for y in range(25)
        ])
        curtailed_annual = np.array([
            np.sum(result[y * self.HOURS_PER_YEAR : (y + 1) * self.HOURS_PER_YEAR])
            for y in range(25)
        ])

        # Year 0: 2% curtailment, Year 24: 2% + 24*1% = 26% curtailment
        rate_year_0 = 1 - curtailed_annual[0] / original_annual[0]
        rate_year_24 = 1 - curtailed_annual[24] / original_annual[24]

        assert rate_year_0 == pytest.approx(0.02, abs=0.001)
        assert rate_year_24 == pytest.approx(0.26, abs=0.001)

    def test_production_correlation(
        self, model: CurtailmentModel
    ) -> None:
        """Test that high production leads to higher curtailment with correlation."""
        # Create production pattern: low in morning, high at midday
        n_years = 5
        n_hours = n_years * self.HOURS_PER_YEAR
        volumes = np.zeros(n_hours)

        for year in range(n_years):
            for day in range(365):
                hour_start = year * self.HOURS_PER_YEAR + day * 24
                for hour_of_day in range(24):
                    # Peak at midday (hour 12): production from 20 to 100
                    production = max(20, 100 - abs(12 - hour_of_day) * 10)
                    volumes[hour_start + hour_of_day] = production

        params = {
            "mode": "stochastic",
            "base_rate": 0.1,  # 10% base rate
            "volatility": 0.0,
            "trend": 0.0,
            "production_correlation": 0.8,  # High correlation
            "seed": 42,
        }

        result = model.apply(volumes, params, n_years=n_years)

        # Calculate total curtailed energy at peak vs off-peak
        peak_original = 0.0
        peak_curtailed_total = 0.0
        offpeak_original = 0.0
        offpeak_curtailed_total = 0.0

        for year in range(n_years):
            for day in range(365):
                hour_start = year * self.HOURS_PER_YEAR + day * 24
                for hour_of_day in range(24):
                    idx = hour_start + hour_of_day
                    if 10 <= hour_of_day <= 14:  # Peak hours
                        peak_original += volumes[idx]
                        peak_curtailed_total += volumes[idx] - result[idx]
                    else:  # Off-peak
                        offpeak_original += volumes[idx]
                        offpeak_curtailed_total += volumes[idx] - result[idx]

        peak_rate = peak_curtailed_total / peak_original
        offpeak_rate = offpeak_curtailed_total / offpeak_original

        # Peak hours should have higher curtailment rate due to correlation
        assert peak_rate > offpeak_rate


class TestCompensation(TestCurtailmentModel):
    """Tests for curtailment compensation calculation (§15 EEG)."""

    def test_compensation_calculation(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test compensation calculation for curtailed energy."""
        n_years = 25

        # Apply 5% curtailment
        params = {"mode": "annual_rates", "rates": 0.05}
        curtailed_volumes = model.apply(hourly_volumes, params, n_years=n_years)

        # Create price array (constant €0.073/kWh)
        hourly_prices = np.ones(n_years * self.HOURS_PER_YEAR) * 0.073

        compensation = model.calculate_compensation(
            original_volumes=hourly_volumes,
            curtailed_volumes=curtailed_volumes,
            hourly_prices=hourly_prices,
            compensation_rate=0.95,
            n_years=n_years,
        )

        # Curtailed energy per year: 100 kWh/h × 8760h × 5% = 43,800 kWh
        expected_curtailed_per_year = 100 * self.HOURS_PER_YEAR * 0.05
        np.testing.assert_allclose(
            compensation["annual_curtailed_kwh"],
            expected_curtailed_per_year,
            rtol=1e-10,
        )

        # Compensation: 43,800 kWh × €0.073/kWh × 95% = ~€3,035/year
        expected_compensation_per_year = expected_curtailed_per_year * 0.073 * 0.95
        np.testing.assert_allclose(
            compensation["annual_compensation_eur"],
            expected_compensation_per_year,
            rtol=1e-10,
        )

        # Total over 25 years
        expected_total = expected_compensation_per_year * 25
        assert compensation["total_compensation_eur"] == pytest.approx(expected_total)

    def test_variable_prices_compensation(
        self, model: CurtailmentModel
    ) -> None:
        """Test compensation with variable prices."""
        n_years = 2
        n_hours = n_years * self.HOURS_PER_YEAR
        volumes = np.ones(n_hours) * 100

        # 10% curtailment
        params = {"mode": "annual_rates", "rates": 0.10}
        curtailed = model.apply(volumes, params, n_years=n_years)

        # Year 0: €0.05/kWh, Year 1: €0.10/kWh
        prices = np.concatenate([
            np.ones(self.HOURS_PER_YEAR) * 0.05,
            np.ones(self.HOURS_PER_YEAR) * 0.10,
        ])

        compensation = model.calculate_compensation(
            original_volumes=volumes,
            curtailed_volumes=curtailed,
            hourly_prices=prices,
            compensation_rate=1.0,  # 100% for easier calculation
            n_years=n_years,
        )

        # Year 0: 100 × 8760 × 10% × €0.05 = €4,380
        # Year 1: 100 × 8760 × 10% × €0.10 = €8,760
        assert compensation["annual_compensation_eur"][0] == pytest.approx(4380.0)
        assert compensation["annual_compensation_eur"][1] == pytest.approx(8760.0)


class TestCurtailmentSummary(TestCurtailmentModel):
    """Tests for curtailment summary statistics."""

    def test_summary_calculation(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test annual summary statistics."""
        n_years = 25
        params = {"mode": "annual_rates", "rates": 0.05}
        curtailed = model.apply(hourly_volumes, params, n_years=n_years)

        stats = model.get_annual_curtailment_summary(
            hourly_volumes, curtailed, n_years=n_years
        )

        # All years should have 5% curtailment
        np.testing.assert_allclose(stats["annual_curtailment_rates"], 0.05, atol=0.001)
        assert stats["total_curtailment_rate"] == pytest.approx(0.05, abs=0.001)

        # Curtailed MWh per year: 100 kWh/h × 8760h × 5% / 1000 = 43.8 MWh
        expected_mwh = 100 * self.HOURS_PER_YEAR * 0.05 / 1000
        np.testing.assert_allclose(stats["annual_curtailed_mwh"], expected_mwh)

    def test_varying_rates_summary(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test summary with varying annual rates."""
        n_years = 5
        volumes = hourly_volumes[:n_years * self.HOURS_PER_YEAR]
        rates = [0.01, 0.02, 0.03, 0.04, 0.05]
        params = {"mode": "annual_rates", "rates": rates}

        curtailed = model.apply(volumes, params, n_years=n_years)
        stats = model.get_annual_curtailment_summary(volumes, curtailed, n_years=n_years)

        # Check each year's rate
        for year, expected_rate in enumerate(rates):
            assert stats["annual_curtailment_rates"][year] == pytest.approx(
                expected_rate, abs=0.001
            )


class TestPerformance(TestCurtailmentModel):
    """Performance tests for curtailment calculations."""

    def test_performance_under_10ms(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Ensure single curtailment calculation takes <10ms."""
        params = {"mode": "annual_rates", "rates": 0.03}

        # Warmup
        model.apply(hourly_volumes, params, n_years=25)

        # Measure 100 iterations
        start = time.perf_counter()
        for _ in range(100):
            model.apply(hourly_volumes, params, n_years=25)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 10, f"Average time {avg_ms:.2f}ms exceeds 10ms target"

    def test_performance_stochastic_mode(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Ensure stochastic mode also meets performance target."""
        params = {
            "mode": "stochastic",
            "base_rate": 0.03,
            "volatility": 0.01,
            "trend": 0.001,
            "production_correlation": 0.3,
            "seed": 42,
        }

        # Warmup
        model.apply(hourly_volumes, params, n_years=25)

        # Measure 100 iterations
        start = time.perf_counter()
        for _ in range(100):
            params["seed"] = 42 + _  # Different seed each time
            model.apply(hourly_volumes, params, n_years=25)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 10, f"Average time {avg_ms:.2f}ms exceeds 10ms target"


class TestEdgeCases(TestCurtailmentModel):
    """Tests for edge cases and error handling."""

    def test_unknown_mode(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test that unknown mode raises error."""
        params = {"mode": "invalid_mode"}

        with pytest.raises(ValueError, match="Unknown curtailment mode"):
            model.apply(hourly_volumes, params, n_years=25)

    def test_zero_production(self, model: CurtailmentModel) -> None:
        """Test with zero production."""
        volumes = np.zeros(100)
        params = {"mode": "annual_rates", "rates": 0.05}

        result = model.apply(volumes, params, n_years=1)
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_100_percent_curtailment(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test with 100% curtailment rate."""
        params = {"mode": "annual_rates", "rates": 1.0}
        result = model.apply(hourly_volumes, params, n_years=25)

        np.testing.assert_array_equal(result, np.zeros_like(hourly_volumes))

    def test_empty_rates_array(
        self, model: CurtailmentModel, hourly_volumes: np.ndarray
    ) -> None:
        """Test with empty rates array defaults to no curtailment."""
        params = {"mode": "annual_rates", "rates": []}
        result = model.apply(hourly_volumes, params, n_years=25)

        # Empty rates should default to no curtailment
        np.testing.assert_array_equal(result, hourly_volumes)

    def test_capacity_limit_below_production(self, model: CurtailmentModel) -> None:
        """Test capacity limit when all production is above limit."""
        volumes = np.ones(100) * 150  # 150 kWh/h

        params = {"mode": "capacity_limit", "limit_factor": 0.50}
        result = model.apply(volumes, params, n_years=1, capacity_kw=100)

        # Max 50 kWh/h
        np.testing.assert_allclose(result, 50.0)
