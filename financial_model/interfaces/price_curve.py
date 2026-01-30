"""Abstract interface for market price curves."""

from abc import ABC, abstractmethod

import numpy as np


class PriceCurveInterface(ABC):
    """
    Abstract interface for market price curves.

    Handles loading and interpolation of electricity/commodity price data.
    The Monte Carlo framework instantiates this with selected scenarios
    ('high', 'mid', 'low').

    CSV Data Format:
        The expected CSV format for hourly price data:

        timestamp,scenario,price_eur_mwh
        2025-01-01 00:00:00,high,85.50
        2025-01-01 00:00:00,mid,72.30
        2025-01-01 00:00:00,low,58.10
        2025-01-01 01:00:00,high,82.40
        ...

        Or wide format:
        timestamp,high,mid,low
        2025-01-01 00:00:00,85.50,72.30,58.10
        2025-01-01 01:00:00,82.40,69.80,55.90
        ...

    Example:
        >>> curve = CSVPriceCurve(curve_type='mid', csv_path='prices.csv')
        >>> hourly = curve.get_hourly_prices(start_year=0, end_year=24)
        >>> len(hourly)
        219000  # 25 years × 8760 hours
    """

    HOURS_PER_YEAR = 8760

    def __init__(self, scenario: str) -> None:
        """
        Initialize with scenario type.

        Args:
            scenario: Scenario identifier ('high', 'mid', 'low').
                Selected by Monte Carlo framework based on simulation parameters.
        """
        if scenario not in ("high", "mid", "low"):
            raise ValueError(
                f"Invalid scenario '{scenario}'. Must be 'high', 'mid', or 'low'."
            )
        self.scenario = scenario

    @abstractmethod
    def get_hourly_prices(self, start_year: int, end_year: int) -> np.ndarray:
        """
        Get hourly prices for specified period.

        Args:
            start_year: First year of the price period (0-indexed project year).
            end_year: Last year of the price period (0-indexed, inclusive).

        Returns:
            np.ndarray of shape (n_hours,) with prices in €/MWh.
            n_hours = (end_year - start_year + 1) × 8760

        Example:
            For a 5-year market period (years 20-24 of a 25-year project):
            >>> curve = CSVPriceCurve(scenario='mid', csv_path='prices.csv')
            >>> prices = curve.get_hourly_prices(20, 24)
            >>> len(prices)
            43800  # 5 years × 8760 hours

        Note:
            - Prices are returned in €/MWh (divide by 1000 for €/kWh)
            - The scenario affects which price column/data is used
            - If requested years exceed available data, prices are extrapolated
        """
        raise NotImplementedError

    def get_monthly_prices(self, start_year: int, end_year: int) -> np.ndarray:
        """
        Get monthly average prices for specified period.

        Aggregates hourly prices to monthly averages for backward compatibility.

        Args:
            start_year: First year of the price period (0-indexed project year).
            end_year: Last year of the price period (0-indexed, inclusive).

        Returns:
            np.ndarray of shape (n_months,) with average prices in €/MWh.
            n_months = (end_year - start_year + 1) × 12
        """
        hourly = self.get_hourly_prices(start_year, end_year)
        n_years = end_year - start_year + 1
        n_months = n_years * 12
        hours_per_month = self.HOURS_PER_YEAR // 12  # 730

        monthly = np.zeros(n_months)
        for month in range(n_months):
            start_hour = month * hours_per_month
            end_hour = start_hour + hours_per_month
            monthly[month] = np.mean(hourly[start_hour:end_hour])

        return monthly

    @property
    def available_years(self) -> int:
        """Return number of years of price data available."""
        raise NotImplementedError
