"""Curtailment model for grid constraints and feed-in management."""

from typing import Any

import numpy as np


class CurtailmentModel:
    """
    Calculates curtailment (feed-in reduction) for energy generation assets.

    Supports multiple curtailment modes:
    - annual_rates: Fixed annual curtailment rates
    - timeseries: CSV-based time-series profiles
    - price_based: Price-threshold curtailment (e.g., §51 EEG negative prices)
    - capacity_limit: Capacity limitation (e.g., 70% rule)
    - stochastic: Stochastic modeling with production correlation for Monte Carlo

    Attributes:
        HOURS_PER_YEAR: Constant for hours per year (8760).
        NEGATIVE_PRICE_CONSECUTIVE_HOURS: Required consecutive hours of negative
            prices before curtailment applies (§51 EEG, reduced to 3 hours).
    """

    HOURS_PER_YEAR = 8760
    NEGATIVE_PRICE_CONSECUTIVE_HOURS = 3  # §51 EEG (updated from 6h to 3h)

    def __init__(self) -> None:
        """Initialize the curtailment model with empty cache."""
        self._cache: dict[str, np.ndarray] = {}

    def apply(
        self,
        volumes: np.ndarray,
        params: dict[str, Any],
        n_years: int,
        hourly_prices: np.ndarray | None = None,
        capacity_kw: float | None = None,
    ) -> np.ndarray:
        """
        Apply curtailment to production volumes.

        Curtailment reduces feed-in based on various factors like grid
        constraints, negative electricity prices, or technical limitations.

        Args:
            volumes: Production array in kWh. Shape: (n_hours,) for hourly
                or (n_months,) for monthly.
            params: Curtailment configuration with at least 'mode' key.
                See mode-specific documentation for required parameters.
            n_years: Project lifetime in years. Required for time-series
                extension and stochastic calculation.
            hourly_prices: Hourly electricity prices in €/kWh. Only required
                for mode='price_based'.
            capacity_kw: Nominal capacity of the asset in kW. Only required
                for mode='capacity_limit'.

        Returns:
            np.ndarray: Curtailed volumes with same shape as input.
                Values are always <= input values.

        Raises:
            ValueError: If required parameters for the chosen mode are
                missing or if an unknown mode is specified.

        Example:
            >>> model = CurtailmentModel()
            >>> volumes = np.ones(25 * 8760) * 100  # 100 kWh/h
            >>> params = {'mode': 'annual_rates', 'rates': 0.03}
            >>> curtailed = model.apply(volumes, params, n_years=25)
            >>> assert np.allclose(curtailed, 97)  # 3% less
        """
        mode = params.get("mode", "annual_rates")

        if mode == "annual_rates":
            return self._apply_annual_rates(volumes, params, n_years)
        elif mode == "timeseries":
            return self._apply_timeseries(volumes, params)
        elif mode == "price_based":
            return self._apply_price_based(volumes, params, hourly_prices, n_years)
        elif mode == "capacity_limit":
            return self._apply_capacity_limit(volumes, params, capacity_kw)
        elif mode == "stochastic":
            return self._apply_stochastic(volumes, params, n_years)
        else:
            raise ValueError(f"Unknown curtailment mode: {mode}")

    def _apply_annual_rates(
        self,
        volumes: np.ndarray,
        params: dict[str, Any],
        n_years: int,
    ) -> np.ndarray:
        """
        Apply fixed annual curtailment rates.

        Args:
            volumes: Production volumes (hourly or monthly).
            params: Must contain 'rates' - single value or array.
            n_years: Project lifetime.

        Returns:
            Curtailed volumes.
        """
        rates = params.get("rates", [])

        # Expand single value to array
        if isinstance(rates, (int, float)):
            rates = [rates] * n_years

        # Extend if needed (repeat last value)
        rates = list(rates)
        while len(rates) < n_years:
            rates.append(rates[-1] if rates else 0.0)

        rates_array = np.array(rates[:n_years])

        # Determine granularity
        is_hourly = len(volumes) == n_years * self.HOURS_PER_YEAR
        is_monthly = len(volumes) == n_years * 12

        if is_hourly:
            # Create hourly factors from annual rates
            factors = np.repeat(1 - rates_array, self.HOURS_PER_YEAR)
        elif is_monthly:
            # Monthly: 12 months per year
            factors = np.repeat(1 - rates_array, 12)
        else:
            # Arbitrary length: apply first rate to entire array
            factors = np.ones(len(volumes)) * (1 - rates_array[0])

        return volumes * factors[: len(volumes)]

    def _apply_timeseries(
        self,
        volumes: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Apply time-series based curtailment profiles from CSV.

        Args:
            volumes: Production volumes.
            params: Must contain 'csv_path' pointing to curtailment profile.

        Returns:
            Curtailed volumes.
        """
        csv_path = params.get("csv_path")
        if csv_path is None:
            raise ValueError("csv_path required for timeseries mode")

        # Caching for performance
        if csv_path not in self._cache:
            self._cache[csv_path] = self._load_curtailment_csv(csv_path)

        factors = self._cache[csv_path]

        # Extend profile to project length (repeat pattern)
        if len(factors) < len(volumes):
            repeats = int(np.ceil(len(volumes) / len(factors)))
            factors = np.tile(factors, repeats)[: len(volumes)]

        return volumes * factors[: len(volumes)]

    def _apply_price_based(
        self,
        volumes: np.ndarray,
        params: dict[str, Any],
        hourly_prices: np.ndarray | None,
        n_years: int,
    ) -> np.ndarray:
        """
        Apply price-based curtailment (e.g., negative prices per §51 EEG).

        Implements the 3-hour rule: Curtailment only applies when prices
        are below threshold for at least 3 consecutive hours (§51 EEG,
        reduced from 6 hours to 3 hours in 2024).

        Args:
            volumes: Production volumes (must be hourly).
            params: Contains 'price_threshold' and 'curtailment_factor'.
            hourly_prices: Hourly electricity prices in €/kWh.
            n_years: Project lifetime.

        Returns:
            Curtailed volumes.
        """
        if hourly_prices is None:
            raise ValueError("hourly_prices required for price_based mode")

        threshold = params.get("price_threshold", 0.0)
        curtailment_factor = params.get("curtailment_factor", 1.0)
        consecutive_hours = params.get(
            "consecutive_hours", self.NEGATIVE_PRICE_CONSECUTIVE_HOURS
        )

        # Ensure prices and volumes align
        n_hours = min(len(volumes), len(hourly_prices))

        # Find hours below threshold
        below_threshold = hourly_prices[:n_hours] < threshold

        # Apply consecutive hours rule (§51 EEG)
        curtail_mask = self._apply_consecutive_hours_rule(
            below_threshold, consecutive_hours
        )

        # Calculate factors
        factors = np.ones(len(volumes), dtype=float)
        factors[:n_hours][curtail_mask] = 1 - curtailment_factor

        return volumes * factors

    def _apply_consecutive_hours_rule(
        self,
        condition_mask: np.ndarray,
        min_consecutive: int,
    ) -> np.ndarray:
        """
        Apply consecutive hours rule (§51 EEG).

        Only marks hours for curtailment if the condition has been true
        for at least min_consecutive hours.

        Args:
            condition_mask: Boolean array indicating condition per hour.
            min_consecutive: Minimum consecutive hours required.

        Returns:
            Boolean mask where True indicates curtailment applies.
        """
        n_hours = len(condition_mask)
        result = np.zeros(n_hours, dtype=bool)

        consecutive_count = 0
        for i in range(n_hours):
            if condition_mask[i]:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    # Mark this hour and all previous consecutive hours
                    for j in range(i - min_consecutive + 1, i + 1):
                        result[j] = True
            else:
                consecutive_count = 0

        return result

    def _apply_capacity_limit(
        self,
        volumes: np.ndarray,
        params: dict[str, Any],
        capacity_kw: float | None,
    ) -> np.ndarray:
        """
        Apply capacity limitation (e.g., 70% rule).

        Args:
            volumes: Production volumes.
            params: Contains 'limit_factor' (e.g., 0.70 for 70%).
            capacity_kw: Nominal capacity of asset in kW.

        Returns:
            Volumes capped at limit_factor × capacity.
        """
        if capacity_kw is None:
            raise ValueError("capacity_kw required for capacity_limit mode")

        limit_factor = params.get("limit_factor", 0.70)
        max_production = capacity_kw * limit_factor

        return np.minimum(volumes, max_production)

    def _apply_stochastic(
        self,
        volumes: np.ndarray,
        params: dict[str, Any],
        n_years: int,
    ) -> np.ndarray:
        """
        Apply stochastic curtailment for Monte Carlo simulations.

        Supports correlation with production: Higher production leads
        to higher curtailment probability (realistic for grid congestion).

        Args:
            volumes: Production volumes.
            params: Contains:
                - base_rate: Base curtailment rate
                - volatility: Standard deviation of rate variation
                - trend: Annual increase in curtailment rate
                - production_correlation: Correlation factor (0-1)
                - seed: Random seed for reproducibility
            n_years: Project lifetime.

        Returns:
            Stochastically curtailed volumes.
        """
        base_rate = params.get("base_rate", 0.02)
        volatility = params.get("volatility", 0.01)
        trend = params.get("trend", 0.0)
        production_correlation = params.get("production_correlation", 0.0)
        seed = params.get("seed")

        if seed is not None:
            np.random.seed(seed)

        is_hourly = len(volumes) == n_years * self.HOURS_PER_YEAR

        # Generate base annual rates with trend and volatility
        years = np.arange(n_years)
        expected_rates = base_rate + trend * years
        random_component = np.random.normal(0, volatility, n_years)
        annual_rates = np.clip(expected_rates + random_component, 0, 1)

        if production_correlation > 0 and is_hourly:
            # Apply production-correlated curtailment at hourly level
            return self._apply_production_correlated_curtailment(
                volumes=volumes,
                annual_rates=annual_rates,
                production_correlation=production_correlation,
                n_years=n_years,
            )
        else:
            # Simple annual rates application
            return self._apply_annual_rates(
                volumes,
                {"rates": annual_rates.tolist()},
                n_years,
            )

    def _apply_production_correlated_curtailment(
        self,
        volumes: np.ndarray,
        annual_rates: np.ndarray,
        production_correlation: float,
        n_years: int,
    ) -> np.ndarray:
        """
        Apply curtailment correlated with production levels.

        Higher production hours are more likely to be curtailed,
        simulating grid congestion during peak generation.

        Args:
            volumes: Hourly production volumes.
            annual_rates: Target annual curtailment rates.
            production_correlation: Strength of correlation (0-1).
            n_years: Project lifetime.

        Returns:
            Curtailed volumes with production correlation.
        """
        result = volumes.copy()

        for year in range(n_years):
            year_start = year * self.HOURS_PER_YEAR
            year_end = year_start + self.HOURS_PER_YEAR
            year_volumes = volumes[year_start:year_end]

            target_rate = annual_rates[year]
            if target_rate <= 0:
                continue

            # Calculate production-weighted curtailment probability
            max_vol = np.max(year_volumes)
            if max_vol <= 0:
                continue

            # Normalized production (0-1)
            normalized = year_volumes / max_vol

            # Base probability from target rate
            # Adjusted by production correlation
            # High production hours get higher curtailment probability
            base_prob = target_rate * (1 - production_correlation)
            correlated_prob = (
                target_rate * production_correlation * 2 * normalized
            )
            hourly_probs = np.clip(base_prob + correlated_prob, 0, 1)

            # Generate curtailment mask
            random_vals = np.random.random(self.HOURS_PER_YEAR)
            curtail_mask = random_vals < hourly_probs

            # Zero out curtailed hours
            result[year_start:year_end][curtail_mask] = 0

        return result

    def _load_curtailment_csv(self, csv_path: str) -> np.ndarray:
        """
        Load curtailment factors from CSV file.

        CSV should have columns: hour/month, curtailment_factor
        Factor of 1.0 means no curtailment, 0.0 means full curtailment.

        Args:
            csv_path: Path to CSV file.

        Returns:
            Array of curtailment factors.
        """
        import csv

        factors = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Accept either 'curtailment_factor' or 'factor'
                factor = float(row.get("curtailment_factor", row.get("factor", 1.0)))
                factors.append(factor)

        return np.array(factors)

    def calculate_compensation(
        self,
        original_volumes: np.ndarray,
        curtailed_volumes: np.ndarray,
        hourly_prices: np.ndarray,
        compensation_rate: float = 0.95,
        n_years: int = 25,
    ) -> dict[str, Any]:
        """
        Calculate curtailment compensation per §15 EEG.

        Under German EEG regulations, operators receive compensation
        for curtailed energy due to grid constraints.

        Args:
            original_volumes: Volumes before curtailment (kWh).
            curtailed_volumes: Volumes after curtailment (kWh).
            hourly_prices: Hourly prices (€/kWh) for compensation calc.
            compensation_rate: Percentage of lost revenue compensated (default 95%).
            n_years: Project lifetime in years.

        Returns:
            Dictionary with:
                - annual_compensation_eur: Annual compensation amounts
                - total_compensation_eur: Total compensation over lifetime
                - annual_curtailed_kwh: Annual curtailed energy
        """
        curtailed_kwh = original_volumes - curtailed_volumes

        is_hourly = len(original_volumes) == n_years * self.HOURS_PER_YEAR

        if is_hourly:
            # Calculate hourly lost revenue
            n_hours = min(len(curtailed_kwh), len(hourly_prices))
            hourly_compensation = (
                curtailed_kwh[:n_hours] * hourly_prices[:n_hours] * compensation_rate
            )

            # Aggregate to annual
            annual_compensation = np.array(
                [
                    np.sum(
                        hourly_compensation[y * self.HOURS_PER_YEAR : (y + 1) * self.HOURS_PER_YEAR]
                    )
                    for y in range(n_years)
                ]
            )

            annual_curtailed = np.array(
                [
                    np.sum(curtailed_kwh[y * self.HOURS_PER_YEAR : (y + 1) * self.HOURS_PER_YEAR])
                    for y in range(n_years)
                ]
            )
        else:
            # Monthly aggregation
            monthly_prices = np.mean(hourly_prices.reshape(-1, 730), axis=1)
            monthly_compensation = curtailed_kwh * monthly_prices[:len(curtailed_kwh)] * compensation_rate

            annual_compensation = np.array(
                [np.sum(monthly_compensation[y * 12 : (y + 1) * 12]) for y in range(n_years)]
            )
            annual_curtailed = np.array(
                [np.sum(curtailed_kwh[y * 12 : (y + 1) * 12]) for y in range(n_years)]
            )

        return {
            "annual_compensation_eur": annual_compensation,
            "total_compensation_eur": float(np.sum(annual_compensation)),
            "annual_curtailed_kwh": annual_curtailed,
        }

    def get_annual_curtailment_summary(
        self,
        original_volumes: np.ndarray,
        curtailed_volumes: np.ndarray,
        n_years: int,
    ) -> dict[str, Any]:
        """
        Calculate annual curtailment statistics.

        Args:
            original_volumes: Production before curtailment.
            curtailed_volumes: Production after curtailment.
            n_years: Project lifetime.

        Returns:
            Dictionary with:
                - annual_curtailed_mwh: MWh curtailed per year
                - annual_curtailment_rates: Rate (0-1) per year
                - total_curtailment_rate: Overall rate
        """
        is_hourly = len(original_volumes) == n_years * self.HOURS_PER_YEAR

        if is_hourly:
            original_annual = np.array(
                [
                    np.sum(original_volumes[y * self.HOURS_PER_YEAR : (y + 1) * self.HOURS_PER_YEAR])
                    for y in range(n_years)
                ]
            )
            curtailed_annual = np.array(
                [
                    np.sum(curtailed_volumes[y * self.HOURS_PER_YEAR : (y + 1) * self.HOURS_PER_YEAR])
                    for y in range(n_years)
                ]
            )
        else:
            original_annual = np.array(
                [np.sum(original_volumes[y * 12 : (y + 1) * 12]) for y in range(n_years)]
            )
            curtailed_annual = np.array(
                [np.sum(curtailed_volumes[y * 12 : (y + 1) * 12]) for y in range(n_years)]
            )

        curtailed_mwh = (original_annual - curtailed_annual) / 1000  # kWh → MWh

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(
                original_annual > 0,
                1 - (curtailed_annual / original_annual),
                0,
            )

        total_original = np.sum(original_volumes)
        total_curtailed = np.sum(curtailed_volumes)
        total_rate = (
            1 - total_curtailed / total_original if total_original > 0 else 0.0
        )

        return {
            "annual_curtailed_mwh": curtailed_mwh,
            "annual_curtailment_rates": rates,
            "total_curtailment_rate": float(total_rate),
        }
