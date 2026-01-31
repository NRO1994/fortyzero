"""PV-specific volume model using PVGIS API for production calculation."""

from typing import Any

import numpy as np

from financial_model.interfaces.volume_model import VolumeModelInterface
from financial_model.utils.pvgis_client import PVGISClient


class PVVolumeModel(VolumeModelInterface):
    """
    PV volume model using PVGIS TMY hourly data.

    Fetches hourly production data from the PVGIS API based on location
    and system parameters, then applies degradation and availability factors
    to generate production forecasts over the project lifetime.

    The model returns hourly production data, enabling integration with
    hourly price curves for accurate revenue calculations.

    Args:
        pvgis_client: Optional PVGISClient instance. Created automatically if None.
        cache_dir: Directory for API response caching.

    Example:
        >>> model = PVVolumeModel()
        >>> params = {
        ...     'capacity': 1000,  # kW
        ...     'volume_model_params': {
        ...         'latitude': 52.52,
        ...         'longitude': 13.405,
        ...         'system_loss': 14,
        ...         'degradation_rate': 0.005,
        ...         'availability': 0.97,
        ...     }
        ... }
        >>> hourly_production = model.calculate(params)
        >>> hourly_production.shape
        (219000,)  # 25 years × 8760 hours
    """

    HOURS_PER_YEAR = 8760

    def __init__(
        self,
        pvgis_client: PVGISClient | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the PV volume model."""
        if pvgis_client is not None:
            self.client = pvgis_client
        else:
            self.client = PVGISClient(cache_dir=cache_dir)

    def calculate(self, params: dict[str, Any]) -> np.ndarray:
        """
        Calculate hourly production volumes over project lifetime.

        Fetches TMY hourly data from PVGIS, scales to installed capacity,
        and applies degradation and availability factors for each year.

        Args:
            params: Technical parameters dictionary containing:
                - capacity: Installed capacity in kW (required)
                - volume_model_params: Dict with:
                    - latitude: Site latitude (required)
                    - longitude: Site longitude (required)
                    - system_loss: System losses in % (default: 14)
                    - tilt_angle: Panel tilt in degrees (default: optimal)
                    - azimuth: Panel azimuth, 0=south (default: 0)
                    - degradation_rate: Annual degradation (default: 0.005)
                    - availability: System availability (default: 0.97)
                    - lifetime_years: Project lifetime (default: 25)

        Returns:
            np.ndarray of shape (n_hours,) with hourly production in kWh.
            n_hours = lifetime_years × 8760

        Raises:
            ValueError: If required parameters are missing.
            PVGISClientError: If API request fails.
        """
        # Extract parameters
        capacity_kw = params.get("capacity")
        if capacity_kw is None:
            raise ValueError("'capacity' parameter is required")

        volume_params = params.get("volume_model_params", {})

        latitude = volume_params.get("latitude")
        longitude = volume_params.get("longitude")

        if latitude is None or longitude is None:
            raise ValueError(
                "'latitude' and 'longitude' are required in volume_model_params"
            )

        system_loss = volume_params.get("system_loss", 14.0)
        tilt_angle = volume_params.get("tilt_angle")  # None = optimal
        azimuth = volume_params.get("azimuth", 0.0)
        degradation_rate = volume_params.get("degradation_rate", 0.005)
        availability = volume_params.get("availability", 0.97)
        lifetime_years = volume_params.get("lifetime_years", 25)

        # Fetch TMY hourly profile from PVGIS (normalized to 1 kWp)
        tmy_data = self.client.get_tmy_hourly(
            lat=latitude,
            lon=longitude,
            peakpower=1.0,  # Normalized to 1 kWp
            loss=system_loss,
            angle=tilt_angle,
            aspect=azimuth,
        )

        # Extract hourly power values and convert to kWh
        hourly_profile = self._extract_hourly_profile(tmy_data)

        # Scale to actual capacity
        hourly_profile_scaled = hourly_profile * capacity_kw

        # Expand to project lifetime with degradation
        hourly_production = self._expand_with_degradation(
            hourly_profile=hourly_profile_scaled,
            degradation_rate=degradation_rate,
            lifetime_years=lifetime_years,
        )

        # Apply availability factor
        hourly_production *= availability

        return hourly_production

    def calculate_monthly(self, params: dict[str, Any]) -> np.ndarray:
        """
        Calculate monthly production volumes (convenience method).

        Aggregates hourly production to monthly totals for use with
        monthly-based financial models.

        Args:
            params: Same as calculate().

        Returns:
            np.ndarray of shape (n_months,) with monthly production in kWh.
        """
        hourly = self.calculate(params)
        return self._aggregate_hourly_to_monthly(hourly)

    def _extract_hourly_profile(self, tmy_data: dict[str, Any]) -> np.ndarray:
        """
        Extract hourly production profile from PVGIS response.

        Args:
            tmy_data: PVGIS TMY response data.

        Returns:
            Array of 8760 hourly production values in kWh (for 1 kWp system).
        """
        hourly_records = tmy_data.get("hourly", [])

        if len(hourly_records) == 0:
            raise ValueError("No hourly data in PVGIS response")

        # PVGIS returns P in Watts for the specified peakpower
        # Since we requested 1 kWp, P is in W per kWp
        # Convert W to kWh (1 hour interval, so W = Wh)
        hourly_values = np.zeros(len(hourly_records))

        for i, record in enumerate(hourly_records):
            # P is power output in W (for 1 kWp system)
            power_w = record.get("P", 0.0)
            # Convert to kWh (power in W for 1 hour = Wh, divide by 1000 for kWh)
            hourly_values[i] = power_w / 1000.0

        # Ensure we have a full year (8760 hours)
        if len(hourly_values) < self.HOURS_PER_YEAR:
            # Pad with zeros if needed (shouldn't happen with valid TMY)
            padded = np.zeros(self.HOURS_PER_YEAR)
            padded[: len(hourly_values)] = hourly_values
            return padded

        return hourly_values[: self.HOURS_PER_YEAR]

    def _expand_with_degradation(
        self,
        hourly_profile: np.ndarray,
        degradation_rate: float,
        lifetime_years: int,
    ) -> np.ndarray:
        """
        Expand single-year profile to full lifetime with degradation.

        Args:
            hourly_profile: Single year hourly profile (8760 values).
            degradation_rate: Annual degradation rate (e.g., 0.005 for 0.5%).
            lifetime_years: Project lifetime in years.

        Returns:
            Array of shape (lifetime_years × 8760,) with degraded production.
        """
        total_hours = lifetime_years * self.HOURS_PER_YEAR
        production = np.zeros(total_hours)

        for year in range(lifetime_years):
            # Degradation factor for this year
            degradation_factor = (1 - degradation_rate) ** year

            # Calculate slice indices
            start_hour = year * self.HOURS_PER_YEAR
            end_hour = start_hour + self.HOURS_PER_YEAR

            # Apply degraded profile
            production[start_hour:end_hour] = hourly_profile * degradation_factor

        return production

    @staticmethod
    def _aggregate_hourly_to_monthly(hourly: np.ndarray) -> np.ndarray:
        """
        Aggregate hourly production to monthly totals.

        Uses a simplified 730-hour month (8760/12) for even distribution.
        For exact monthly totals, use actual month lengths.

        Args:
            hourly: Hourly production array.

        Returns:
            Monthly production array.
        """
        # Hours per month (approximate, using 365.25 days / 12)
        hours_per_month = 730

        n_months = len(hourly) // hours_per_month
        monthly = np.zeros(n_months)

        for month in range(n_months):
            start = month * hours_per_month
            end = start + hours_per_month
            monthly[month] = np.sum(hourly[start:end])

        # Handle remaining hours (leap year adjustment)
        remaining_hours = len(hourly) - (n_months * hours_per_month)
        if remaining_hours > 0:
            monthly[-1] += np.sum(hourly[-(remaining_hours):])

        return monthly

    def get_annual_production(self, params: dict[str, Any]) -> np.ndarray:
        """
        Get annual production totals.

        Args:
            params: Technical parameters.

        Returns:
            Array of annual production values in kWh.
        """
        hourly = self.calculate(params)
        n_years = len(hourly) // self.HOURS_PER_YEAR
        annual = np.zeros(n_years)

        for year in range(n_years):
            start = year * self.HOURS_PER_YEAR
            end = start + self.HOURS_PER_YEAR
            annual[year] = np.sum(hourly[start:end])

        return annual
