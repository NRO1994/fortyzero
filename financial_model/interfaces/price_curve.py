"""Abstract interface for market price curves."""

from abc import ABC, abstractmethod

import numpy as np


class PriceCurveInterface(ABC):
    """
    Abstract interface for market price curves.

    Handles loading and interpolation of electricity/commodity price data.
    The Monte Carlo framework instantiates this with selected scenarios
    ('high', 'mid', 'low').

    Example implementation:
        >>> class CSVPriceCurve(PriceCurveInterface):
        ...     def __init__(self, curve_type: str, csv_path: str):
        ...         super().__init__(curve_type)
        ...         self.data = pd.read_csv(csv_path)
        ...         self.curve_type = curve_type
        ...
        ...     def get_monthly_prices(self, start_year, end_year):
        ...         # Filter and aggregate hourly data to monthly
        ...         filtered = self.data[
        ...             (self.data['year'] >= start_year) &
        ...             (self.data['year'] <= end_year) &
        ...             (self.data['scenario'] == self.curve_type)
        ...         ]
        ...         return filtered.groupby('month')['price'].mean().values
    """

    def __init__(self, curve_type: str) -> None:
        """
        Initialize with curve type.

        Args:
            curve_type: Scenario identifier ('high', 'mid', 'low').
                Selected by Monte Carlo framework based on simulation parameters.
        """
        self.curve_type = curve_type

    @abstractmethod
    def get_monthly_prices(self, start_year: int, end_year: int) -> np.ndarray:
        """
        Get monthly average prices for specified period.

        Args:
            start_year: First year of the price period (0-indexed project year).
            end_year: Last year of the price period (0-indexed project year).

        Returns:
            np.ndarray of shape (n_months,) with prices in €/kWh or €/MWh.
            Array should be aligned with project timeline, starting from
            start_year * 12 and ending at (end_year + 1) * 12.

        Example:
            For a 5-year market period (years 20-24 of a 25-year project):
            >>> curve = SomePriceCurve(curve_type='mid')
            >>> prices = curve.get_monthly_prices(20, 24)
            >>> len(prices)
            60  # 5 years × 12 months

        Note:
            - Prices should account for seasonal patterns if applicable
            - The curve_type affects which scenario data is used
            - Implementation may load from CSV, database, or API
        """
        raise NotImplementedError
