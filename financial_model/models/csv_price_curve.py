"""CSV-based price curve implementation for hourly electricity prices."""

from pathlib import Path
from typing import Any

import numpy as np

from financial_model.interfaces.price_curve import PriceCurveInterface


class CSVPriceCurve(PriceCurveInterface):
    """
    Price curve implementation that reads hourly prices from CSV files.

    Supports CSV files with wide format (scenario columns) or long format.
    Handles year mapping from project years to data years, with extrapolation
    for years beyond available data.

    Args:
        scenario: Price scenario ('high', 'mid', 'low').
        csv_path: Path to CSV file with hourly price data.
        price_column: Column name for prices (only for long format).
        scenario_column: Column name for scenario (only for long format).

    Expected CSV format (wide, recommended):
        timestamp,high,mid,low
        2025-01-01 00:00:00,85.50,72.30,58.10
        2025-01-01 01:00:00,82.40,69.80,55.90
        ...

    Alternative CSV format (long):
        timestamp,scenario,price_eur_mwh
        2025-01-01 00:00:00,high,85.50
        2025-01-01 00:00:00,mid,72.30
        ...

    Example:
        >>> curve = CSVPriceCurve(scenario='mid', csv_path='prices.csv')
        >>> prices = curve.get_hourly_prices(start_year=0, end_year=24)
        >>> len(prices)
        219000
    """

    def __init__(
        self,
        scenario: str,
        csv_path: str | Path,
        price_column: str | None = None,
        scenario_column: str | None = None,
    ) -> None:
        """Initialize the CSV price curve."""
        super().__init__(scenario)

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Price CSV not found: {self.csv_path}")

        self.price_column = price_column
        self.scenario_column = scenario_column

        # Load and cache data
        self._prices: np.ndarray | None = None
        self._data_years: int = 0
        self._load_data()

    def _load_data(self) -> None:
        """Load price data from CSV file."""
        # Use numpy for fast loading (skip header)
        # First, detect format by reading header
        with open(self.csv_path, "r") as f:
            header = f.readline().strip().split(",")

        if self.scenario in header:
            # Wide format: timestamp,high,mid,low
            self._load_wide_format(header)
        elif self.scenario_column and self.scenario_column in header:
            # Long format: timestamp,scenario,price
            self._load_long_format()
        else:
            # Try to detect
            if "high" in header or "mid" in header or "low" in header:
                self._load_wide_format(header)
            else:
                raise ValueError(
                    f"Cannot detect CSV format. Expected columns: "
                    f"'high', 'mid', 'low' (wide) or specify scenario_column."
                )

    def _load_wide_format(self, header: list[str]) -> None:
        """Load wide format CSV (scenario as columns)."""
        # Find column index for our scenario
        try:
            col_idx = header.index(self.scenario)
        except ValueError:
            raise ValueError(
                f"Scenario '{self.scenario}' not found in CSV columns: {header}"
            )

        # Load just the price column using numpy for speed
        # Skip header row, load only the scenario column
        try:
            import pandas as pd

            df = pd.read_csv(self.csv_path, usecols=[self.scenario])
            self._prices = df[self.scenario].values.astype(np.float64)
        except ImportError:
            # Fallback without pandas (slower)
            self._prices = np.loadtxt(
                self.csv_path,
                delimiter=",",
                skiprows=1,
                usecols=[col_idx],
                dtype=np.float64,
            )

        self._data_years = len(self._prices) // self.HOURS_PER_YEAR

    def _load_long_format(self) -> None:
        """Load long format CSV (scenario as rows)."""
        import pandas as pd

        df = pd.read_csv(self.csv_path)

        scenario_col = self.scenario_column or "scenario"
        price_col = self.price_column or "price_eur_mwh"

        # Filter to our scenario
        mask = df[scenario_col] == self.scenario
        filtered = df.loc[mask, price_col]

        self._prices = filtered.values.astype(np.float64)
        self._data_years = len(self._prices) // self.HOURS_PER_YEAR

    def get_hourly_prices(self, start_year: int, end_year: int) -> np.ndarray:
        """
        Get hourly prices for specified period.

        Args:
            start_year: First year (0-indexed project year).
            end_year: Last year (0-indexed, inclusive).

        Returns:
            Array of hourly prices in €/MWh.
        """
        if self._prices is None:
            raise RuntimeError("Price data not loaded")

        n_years = end_year - start_year + 1
        n_hours = n_years * self.HOURS_PER_YEAR
        result = np.zeros(n_hours)

        for year in range(n_years):
            project_year = start_year + year
            start_hour = year * self.HOURS_PER_YEAR
            end_hour = start_hour + self.HOURS_PER_YEAR

            # Get prices for this year (with extrapolation if needed)
            year_prices = self._get_year_prices(project_year)
            result[start_hour:end_hour] = year_prices

        return result

    def _get_year_prices(self, project_year: int) -> np.ndarray:
        """
        Get hourly prices for a single project year.

        If project_year exceeds available data, extrapolates using the last
        available year with optional trend adjustment.

        Args:
            project_year: Project year (0-indexed).

        Returns:
            Array of 8760 hourly prices.
        """
        if project_year < self._data_years:
            # Direct mapping: use data for this year
            start_idx = project_year * self.HOURS_PER_YEAR
            end_idx = start_idx + self.HOURS_PER_YEAR
            return self._prices[start_idx:end_idx].copy()
        else:
            # Extrapolation: use last available year
            # Apply simple trend based on average growth in data
            last_year_idx = self._data_years - 1
            start_idx = last_year_idx * self.HOURS_PER_YEAR
            end_idx = start_idx + self.HOURS_PER_YEAR
            base_prices = self._prices[start_idx:end_idx].copy()

            # Calculate years beyond data
            years_beyond = project_year - last_year_idx

            # Apply 2% annual escalation for extrapolation
            escalation_factor = (1.02) ** years_beyond
            return base_prices * escalation_factor

    @property
    def available_years(self) -> int:
        """Return number of years of price data available."""
        return self._data_years

    def get_price_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics of loaded price data.

        Returns:
            Dictionary with min, max, mean, std of prices.
        """
        if self._prices is None:
            return {}

        return {
            "scenario": self.scenario,
            "years": self._data_years,
            "hours": len(self._prices),
            "min": float(np.min(self._prices)),
            "max": float(np.max(self._prices)),
            "mean": float(np.mean(self._prices)),
            "std": float(np.std(self._prices)),
        }
