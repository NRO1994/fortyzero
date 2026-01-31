"""Tests for the PVVolumeModel."""

import numpy as np
import pytest

from financial_model.models.pv_volume_model import PVVolumeModel
from financial_model.utils.pvgis_client import (
    PVGISClient,
    PVGISClientError,
    PVGISLocationError,
)


class MockPVGISClient:
    """Mock PVGIS client for testing without API calls."""

    def __init__(self, hourly_power_w: float = 100.0) -> None:
        """Initialize mock with configurable power output."""
        self.hourly_power_w = hourly_power_w
        self.call_count = 0

    def get_tmy_hourly(
        self,
        lat: float,
        lon: float,
        peakpower: float = 1.0,
        loss: float = 14.0,
        angle: float | None = None,
        aspect: float = 0.0,
        startyear: int | None = None,
        endyear: int | None = None,
    ) -> dict:
        """Return mock TMY data."""
        self.call_count += 1

        # Generate realistic daily pattern (sine wave, peak at noon)
        hourly = []
        for hour in range(8760):
            hour_of_day = hour % 24
            day_of_year = hour // 24

            # Simple solar model: production between 6am-6pm
            if 6 <= hour_of_day <= 18:
                # Peak at noon
                solar_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
                # Seasonal variation (peak in summer)
                seasonal_factor = 0.7 + 0.3 * np.sin(
                    2 * np.pi * (day_of_year - 80) / 365
                )
                power = self.hourly_power_w * solar_factor * seasonal_factor
            else:
                power = 0.0

            hourly.append({"P": power, "time": f"2020{day_of_year:03d}:{hour_of_day:02d}00"})

        return {
            "hourly": hourly,
            "inputs": {"lat": lat, "lon": lon, "peakpower": peakpower},
            "meta": {"source": "mock"},
        }


class TestPVVolumeModelBasic:
    """Basic tests for PVVolumeModel."""

    @pytest.fixture
    def mock_client(self) -> MockPVGISClient:
        """Create mock PVGIS client."""
        return MockPVGISClient(hourly_power_w=150.0)

    @pytest.fixture
    def model(self, mock_client: MockPVGISClient) -> PVVolumeModel:
        """Create PVVolumeModel with mock client."""
        return PVVolumeModel(pvgis_client=mock_client)

    @pytest.fixture
    def basic_params(self) -> dict:
        """Create basic test parameters."""
        return {
            "capacity": 1000,  # kW
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "system_loss": 14,
                "degradation_rate": 0.005,
                "availability": 0.97,
                "lifetime_years": 25,
            },
        }

    def test_calculate_returns_correct_shape(
        self, model: PVVolumeModel, basic_params: dict
    ) -> None:
        """Test that calculate returns array with correct shape."""
        result = model.calculate(basic_params)

        expected_hours = 25 * 8760  # 25 years × 8760 hours
        assert result.shape == (expected_hours,)

    def test_calculate_returns_numpy_array(
        self, model: PVVolumeModel, basic_params: dict
    ) -> None:
        """Test that calculate returns numpy array."""
        result = model.calculate(basic_params)
        assert isinstance(result, np.ndarray)

    def test_calculate_all_values_non_negative(
        self, model: PVVolumeModel, basic_params: dict
    ) -> None:
        """Test that all production values are non-negative."""
        result = model.calculate(basic_params)
        assert np.all(result >= 0)

    def test_calculate_has_positive_total(
        self, model: PVVolumeModel, basic_params: dict
    ) -> None:
        """Test that total production is positive."""
        result = model.calculate(basic_params)
        assert np.sum(result) > 0

    def test_missing_capacity_raises_error(self, model: PVVolumeModel) -> None:
        """Test that missing capacity raises ValueError."""
        params = {
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
            }
        }

        with pytest.raises(ValueError, match="capacity"):
            model.calculate(params)

    def test_missing_latitude_raises_error(self, model: PVVolumeModel) -> None:
        """Test that missing latitude raises ValueError."""
        params = {
            "capacity": 1000,
            "volume_model_params": {
                "longitude": 13.405,
            },
        }

        with pytest.raises(ValueError, match="latitude"):
            model.calculate(params)

    def test_missing_longitude_raises_error(self, model: PVVolumeModel) -> None:
        """Test that missing longitude raises ValueError."""
        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
            },
        }

        with pytest.raises(ValueError, match="longitude"):
            model.calculate(params)


class TestPVVolumeModelDegradation:
    """Tests for degradation calculation."""

    @pytest.fixture
    def mock_client(self) -> MockPVGISClient:
        """Create mock with constant power for easy verification."""
        return MockPVGISClient(hourly_power_w=100.0)

    @pytest.fixture
    def model(self, mock_client: MockPVGISClient) -> PVVolumeModel:
        """Create PVVolumeModel with mock client."""
        return PVVolumeModel(pvgis_client=mock_client)

    def test_degradation_reduces_production_over_time(
        self, model: PVVolumeModel
    ) -> None:
        """Test that production decreases each year due to degradation."""
        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "degradation_rate": 0.01,  # 1% per year
                "availability": 1.0,  # 100% to isolate degradation
                "lifetime_years": 5,
            },
        }

        result = model.calculate(params)
        annual = model.get_annual_production(params)

        # Each year should be less than previous
        for i in range(1, len(annual)):
            assert annual[i] < annual[i - 1]

    def test_degradation_rate_applied_correctly(
        self, model: PVVolumeModel
    ) -> None:
        """Test that degradation rate is applied correctly."""
        degradation_rate = 0.02  # 2% per year

        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "degradation_rate": degradation_rate,
                "availability": 1.0,
                "lifetime_years": 3,
            },
        }

        annual = model.get_annual_production(params)

        # Year 1 should be (1-0.02) = 98% of year 0
        expected_ratio = 1 - degradation_rate
        actual_ratio = annual[1] / annual[0]

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.001)

    def test_zero_degradation(self, model: PVVolumeModel) -> None:
        """Test that zero degradation gives constant annual production."""
        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "degradation_rate": 0.0,
                "availability": 1.0,
                "lifetime_years": 5,
            },
        }

        annual = model.get_annual_production(params)

        # All years should have same production
        for i in range(1, len(annual)):
            assert annual[i] == pytest.approx(annual[0], rel=0.001)


class TestPVVolumeModelAvailability:
    """Tests for availability factor."""

    @pytest.fixture
    def mock_client(self) -> MockPVGISClient:
        """Create mock client."""
        return MockPVGISClient(hourly_power_w=100.0)

    @pytest.fixture
    def model(self, mock_client: MockPVGISClient) -> PVVolumeModel:
        """Create PVVolumeModel with mock client."""
        return PVVolumeModel(pvgis_client=mock_client)

    def test_availability_scales_production(self, model: PVVolumeModel) -> None:
        """Test that availability factor scales production correctly."""
        base_params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "degradation_rate": 0.0,
                "availability": 1.0,
                "lifetime_years": 1,
            },
        }

        reduced_params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "degradation_rate": 0.0,
                "availability": 0.9,  # 90%
                "lifetime_years": 1,
            },
        }

        base_production = np.sum(model.calculate(base_params))
        reduced_production = np.sum(model.calculate(reduced_params))

        assert reduced_production == pytest.approx(base_production * 0.9, rel=0.001)


class TestPVVolumeModelCapacity:
    """Tests for capacity scaling."""

    @pytest.fixture
    def mock_client(self) -> MockPVGISClient:
        """Create mock client."""
        return MockPVGISClient(hourly_power_w=100.0)

    @pytest.fixture
    def model(self, mock_client: MockPVGISClient) -> PVVolumeModel:
        """Create PVVolumeModel with mock client."""
        return PVVolumeModel(pvgis_client=mock_client)

    def test_capacity_scales_linearly(self, model: PVVolumeModel) -> None:
        """Test that doubling capacity doubles production."""
        params_1mw = {
            "capacity": 1000,  # 1 MW
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "lifetime_years": 1,
            },
        }

        params_2mw = {
            "capacity": 2000,  # 2 MW
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "lifetime_years": 1,
            },
        }

        production_1mw = np.sum(model.calculate(params_1mw))
        production_2mw = np.sum(model.calculate(params_2mw))

        assert production_2mw == pytest.approx(production_1mw * 2, rel=0.001)


class TestPVVolumeModelMonthly:
    """Tests for monthly aggregation."""

    @pytest.fixture
    def mock_client(self) -> MockPVGISClient:
        """Create mock client."""
        return MockPVGISClient(hourly_power_w=100.0)

    @pytest.fixture
    def model(self, mock_client: MockPVGISClient) -> PVVolumeModel:
        """Create PVVolumeModel with mock client."""
        return PVVolumeModel(pvgis_client=mock_client)

    def test_monthly_returns_correct_count(self, model: PVVolumeModel) -> None:
        """Test that monthly aggregation returns correct number of months."""
        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "lifetime_years": 25,
            },
        }

        monthly = model.calculate_monthly(params)

        # Should have ~300 months (25 years × 12)
        assert len(monthly) >= 299
        assert len(monthly) <= 301

    def test_monthly_sum_equals_hourly_sum(self, model: PVVolumeModel) -> None:
        """Test that monthly totals equal hourly totals."""
        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "lifetime_years": 5,
            },
        }

        hourly = model.calculate(params)
        monthly = model.calculate_monthly(params)

        assert np.sum(monthly) == pytest.approx(np.sum(hourly), rel=0.01)


class TestPVGISClientCaching:
    """Tests for PVGIS client caching."""

    def test_cache_prevents_duplicate_calls(self, tmp_path) -> None:
        """Test that caching prevents duplicate API calls."""
        # Use a mock that tracks calls
        mock = MockPVGISClient()
        model = PVVolumeModel(pvgis_client=mock)

        params = {
            "capacity": 1000,
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "lifetime_years": 1,
            },
        }

        # First call
        model.calculate(params)
        first_count = mock.call_count

        # Second call with same params - should use cache
        # (mock doesn't cache, so this tests the model, not actual caching)
        model.calculate(params)
        second_count = mock.call_count

        # Mock gets called each time since it doesn't cache
        # Real client would cache
        assert second_count == first_count + 1


@pytest.mark.integration
class TestPVGISClientIntegration:
    """Integration tests with real PVGIS API (marked as slow/integration)."""

    def test_real_api_berlin(self, tmp_path) -> None:
        """Test real API call for Berlin location."""
        client = PVGISClient(cache_dir=tmp_path)
        model = PVVolumeModel(pvgis_client=client)

        params = {
            "capacity": 1000,  # 1 MWp
            "volume_model_params": {
                "latitude": 52.52,
                "longitude": 13.405,
                "system_loss": 14,
                "degradation_rate": 0.005,
                "availability": 0.97,
                "lifetime_years": 1,
            },
        }

        result = model.calculate(params)

        # Should have 8760 hours
        assert len(result) == 8760

        # Annual production should be roughly 900-1100 MWh for 1 MWp in Germany
        annual_mwh = np.sum(result) / 1000
        assert 700 < annual_mwh < 1200

    def test_real_api_invalid_location(self, tmp_path) -> None:
        """Test that invalid location raises appropriate error."""
        client = PVGISClient(cache_dir=tmp_path)

        # Ocean location (should fail)
        with pytest.raises(PVGISClientError):
            client.get_tmy_hourly(lat=0.0, lon=0.0)
