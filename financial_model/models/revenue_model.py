"""Revenue model for calculating monthly revenues from multiple streams."""

from typing import Any

import numpy as np

from financial_model.interfaces.price_curve import PriceCurveInterface


class RevenueModel:
    """
    Calculates monthly revenues from configured revenue streams.

    Supports fixed-price periods (e.g., EEG tariff) and market-based
    periods with price curve integration.
    """

    def calculate(
        self,
        volumes: np.ndarray,
        revenue_config: dict[str, Any],
        inflation_rate: float,
        price_curve: PriceCurveInterface | None,
        n_months: int,
    ) -> np.ndarray:
        """
        Calculate monthly revenues across all streams.

        Args:
            volumes: Monthly production volumes.
            revenue_config: Revenue configuration with streams list.
            inflation_rate: Base inflation rate for escalation.
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
                inflation_rate=inflation_rate,
                price_curve=price_curve,
                n_months=n_months,
            )
            monthly_revenue += stream_revenue

        return monthly_revenue

    def _calculate_stream_revenue(
        self,
        volumes: np.ndarray,
        stream: dict[str, Any],
        inflation_rate: float,
        price_curve: PriceCurveInterface | None,
        n_months: int,
    ) -> np.ndarray:
        """
        Calculate revenue for a single stream.

        Args:
            volumes: Monthly production volumes.
            stream: Stream configuration.
            inflation_rate: Base inflation rate.
            price_curve: Optional price curve interface.
            n_months: Total months.

        Returns:
            Monthly revenue array for this stream.
        """
        revenue = np.zeros(n_months)
        price_structure = stream.get("price_structure", {})

        for month in range(n_months):
            year = month // 12

            price = self._get_price_for_month(
                year=year,
                month=month,
                price_structure=price_structure,
                inflation_rate=inflation_rate,
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
        inflation_rate: float,
        price_curve: PriceCurveInterface | None,
    ) -> float | None:
        """
        Determine the price for a specific month.

        Args:
            year: Year index (0-based).
            month: Month index (0-based).
            price_structure: Price structure configuration.
            inflation_rate: Base inflation rate.
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
                    escalation = fixed_period.get("escalation_rate") or inflation_rate
                    return base_price * (1 + escalation) ** year
                return base_price

        # Check market period
        market_period = price_structure.get("market_period", {})
        if market_period and price_curve is not None:
            start_year = market_period.get("start_year", 0)
            end_year = market_period.get("end_year", float("inf"))

            if start_year <= year <= end_year:
                monthly_prices = price_curve.get_monthly_prices(start_year, end_year)
                # Adjust index to market period start
                market_month = month - (start_year * 12)
                if 0 <= market_month < len(monthly_prices):
                    return monthly_prices[market_month]

        return None
