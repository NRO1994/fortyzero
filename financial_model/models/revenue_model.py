"""Revenue model for calculating hourly/monthly revenues from multiple streams."""

from typing import Any

import numpy as np

from financial_model.interfaces.price_curve import PriceCurveInterface


class RevenueModel:
    """
    Calculates revenues from configured revenue streams.

    Supports both hourly and monthly granularity:
    - Hourly: For detailed analysis with hourly price curves
    - Monthly: For backward compatibility and faster calculations

    Handles fixed-price periods (e.g., EEG tariff) and market-based
    periods with price curve integration.
    """

    HOURS_PER_YEAR = 8760
    HOURS_PER_MONTH = 730  # Approximate

    def calculate_hourly(
        self,
        hourly_volumes: np.ndarray,
        revenue_config: dict[str, Any],
        inflation_rates: np.ndarray,
        price_curve: PriceCurveInterface | None,
        n_years: int,
    ) -> np.ndarray:
        """
        Calculate hourly revenues across all streams.

        Args:
            hourly_volumes: Hourly production volumes (n_years × 8760 values).
            revenue_config: Revenue configuration with streams list.
            inflation_rates: Year-indexed inflation rates array of shape (n_years,).
            price_curve: Price curve for market periods.
            n_years: Project lifetime in years.

        Returns:
            Array of hourly revenues in € (same shape as hourly_volumes).
        """
        n_hours = n_years * self.HOURS_PER_YEAR

        if len(hourly_volumes) != n_hours:
            raise ValueError(
                f"Expected {n_hours} hourly volumes, got {len(hourly_volumes)}"
            )

        hourly_revenue = np.zeros(n_hours)

        for stream in revenue_config.get("streams", []):
            stream_revenue = self._calculate_stream_hourly(
                hourly_volumes=hourly_volumes,
                stream=stream,
                inflation_rates=inflation_rates,
                price_curve=price_curve,
                n_years=n_years,
            )
            hourly_revenue += stream_revenue

        return hourly_revenue

    def _calculate_stream_hourly(
        self,
        hourly_volumes: np.ndarray,
        stream: dict[str, Any],
        inflation_rates: np.ndarray,
        price_curve: PriceCurveInterface | None,
        n_years: int,
    ) -> np.ndarray:
        """
        Calculate hourly revenue for a single stream.

        Args:
            hourly_volumes: Hourly production volumes.
            stream: Stream configuration.
            inflation_rates: Year-indexed inflation rates array.
            price_curve: Optional price curve interface.
            n_years: Project lifetime.

        Returns:
            Hourly revenue array for this stream.
        """
        n_hours = n_years * self.HOURS_PER_YEAR
        revenue = np.zeros(n_hours)
        price_structure = stream.get("price_structure", {})

        # Get fixed period parameters
        fixed_period = price_structure.get("fixed_period", {})
        fixed_start = fixed_period.get("start_year", 0)
        fixed_end = fixed_period.get("end_year", -1)
        fixed_price = fixed_period.get("price", 0.0)  # €/kWh
        fixed_indexed = fixed_period.get("indexed", False)
        custom_escalation = fixed_period.get("escalation_rate")

        # Pre-calculate cumulative factors for time-varying inflation
        cumulative_factors = np.cumprod(1 + inflation_rates)
        cumulative_factors = np.insert(cumulative_factors[:-1], 0, 1.0)

        # Get market period parameters
        market_period = price_structure.get("market_period", {})
        market_start = market_period.get("start_year", n_years)
        market_end = market_period.get("end_year", n_years - 1)

        # Pre-calculate hourly prices for market period if applicable
        market_prices = None
        if price_curve is not None and market_start <= n_years - 1:
            actual_market_end = min(market_end, n_years - 1)
            market_prices = price_curve.get_hourly_prices(market_start, actual_market_end)
            # Convert €/MWh to €/kWh
            market_prices = market_prices / 1000.0

        # Calculate revenue for each hour using vectorization where possible
        for year in range(n_years):
            year_start_hour = year * self.HOURS_PER_YEAR
            year_end_hour = year_start_hour + self.HOURS_PER_YEAR
            year_volumes = hourly_volumes[year_start_hour:year_end_hour]

            if fixed_start <= year <= fixed_end:
                # Fixed price period
                if fixed_indexed:
                    if custom_escalation is not None:
                        # Use constant custom escalation rate
                        price = fixed_price * (1 + custom_escalation) ** year
                    else:
                        # Use time-varying inflation via cumulative factors
                        price = fixed_price * cumulative_factors[year]
                else:
                    price = fixed_price

                revenue[year_start_hour:year_end_hour] = year_volumes * price

            elif market_start <= year <= market_end and market_prices is not None:
                # Market price period
                market_year_idx = year - market_start
                market_hour_start = market_year_idx * self.HOURS_PER_YEAR
                market_hour_end = market_hour_start + self.HOURS_PER_YEAR

                if market_hour_end <= len(market_prices):
                    year_prices = market_prices[market_hour_start:market_hour_end]
                    revenue[year_start_hour:year_end_hour] = year_volumes * year_prices

        return revenue

    def calculate(
        self,
        volumes: np.ndarray,
        revenue_config: dict[str, Any],
        inflation_rates: np.ndarray,
        price_curve: PriceCurveInterface | None,
        n_months: int,
    ) -> np.ndarray:
        """
        Calculate monthly revenues across all streams.

        Backward-compatible method that works with monthly volumes.

        Args:
            volumes: Monthly production volumes.
            revenue_config: Revenue configuration with streams list.
            inflation_rates: Year-indexed inflation rates array.
            price_curve: Optional price curve for market periods.
            n_months: Total months in project lifetime.

        Returns:
            Array of total monthly revenues.
        """
        monthly_revenue = np.zeros(n_months)

        for stream in revenue_config.get("streams", []):
            stream_revenue = self._calculate_stream_revenue(
                volumes=volumes,
                stream=stream,
                inflation_rates=inflation_rates,
                price_curve=price_curve,
                n_months=n_months,
            )
            monthly_revenue += stream_revenue

        return monthly_revenue

    def _calculate_stream_revenue(
        self,
        volumes: np.ndarray,
        stream: dict[str, Any],
        inflation_rates: np.ndarray,
        price_curve: PriceCurveInterface | None,
        n_months: int,
    ) -> np.ndarray:
        """
        Calculate revenue for a single stream (monthly granularity).

        Args:
            volumes: Monthly production volumes.
            stream: Stream configuration.
            inflation_rates: Year-indexed inflation rates array.
            price_curve: Optional price curve interface.
            n_months: Total months.

        Returns:
            Monthly revenue array for this stream.
        """
        revenue = np.zeros(n_months)
        price_structure = stream.get("price_structure", {})

        # Pre-calculate cumulative factors for time-varying inflation
        cumulative_factors = np.cumprod(1 + inflation_rates)
        cumulative_factors = np.insert(cumulative_factors[:-1], 0, 1.0)

        for month in range(n_months):
            year = month // 12

            price = self._get_price_for_month(
                year=year,
                month=month,
                price_structure=price_structure,
                cumulative_factors=cumulative_factors,
                price_curve=price_curve,
            )

            if price is not None:
                revenue[month] = volumes[month] * price

        return revenue

    def _get_price_for_month(
        self,
        year: int,
        month: int,
        price_structure: dict[str, Any],
        cumulative_factors: np.ndarray,
        price_curve: PriceCurveInterface | None,
    ) -> float | None:
        """
        Determine the price for a specific month.

        Args:
            year: Year index (0-based).
            month: Month index (0-based).
            price_structure: Price structure configuration.
            cumulative_factors: Pre-calculated cumulative inflation factors.
            price_curve: Optional price curve interface.

        Returns:
            Price for the month, or None if outside all periods.
        """
        # Check fixed period
        fixed_period = price_structure.get("fixed_period", {})
        if fixed_period:
            start_year = fixed_period.get("start_year", 0)
            end_year = fixed_period.get("end_year", float("inf"))

            if start_year <= year <= end_year:
                base_price = fixed_period["price"]

                if fixed_period.get("indexed", False):
                    custom_escalation = fixed_period.get("escalation_rate")
                    if custom_escalation is not None:
                        # Use constant custom escalation rate
                        return base_price * (1 + custom_escalation) ** year
                    else:
                        # Use time-varying inflation via cumulative factors
                        return base_price * cumulative_factors[year]
                return base_price

        # Check market period
        market_period = price_structure.get("market_period", {})
        if market_period and price_curve is not None:
            start_year = market_period.get("start_year", 0)
            end_year = market_period.get("end_year", float("inf"))

            if start_year <= year <= end_year:
                monthly_prices = price_curve.get_monthly_prices(start_year, int(end_year))
                # Adjust index to market period start
                market_month = month - (start_year * 12)
                if 0 <= market_month < len(monthly_prices):
                    # Convert €/MWh to €/kWh
                    return monthly_prices[market_month] / 1000.0

        return None

    @staticmethod
    def aggregate_hourly_to_monthly(hourly_revenue: np.ndarray) -> np.ndarray:
        """
        Aggregate hourly revenues to monthly totals.

        Args:
            hourly_revenue: Hourly revenue array.

        Returns:
            Monthly revenue array.
        """
        n_hours = len(hourly_revenue)
        hours_per_month = 730
        n_months = n_hours // hours_per_month

        monthly = np.zeros(n_months)
        for month in range(n_months):
            start = month * hours_per_month
            end = start + hours_per_month
            monthly[month] = np.sum(hourly_revenue[start:end])

        # Handle remaining hours
        remaining = n_hours - (n_months * hours_per_month)
        if remaining > 0:
            monthly[-1] += np.sum(hourly_revenue[-remaining:])

        return monthly

    @staticmethod
    def aggregate_hourly_to_annual(hourly_revenue: np.ndarray) -> np.ndarray:
        """
        Aggregate hourly revenues to annual totals.

        Args:
            hourly_revenue: Hourly revenue array.

        Returns:
            Annual revenue array.
        """
        hours_per_year = 8760
        n_years = len(hourly_revenue) // hours_per_year

        annual = np.zeros(n_years)
        for year in range(n_years):
            start = year * hours_per_year
            end = start + hours_per_year
            annual[year] = np.sum(hourly_revenue[start:end])

        return annual
