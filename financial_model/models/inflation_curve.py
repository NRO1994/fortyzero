"""Inflation curve loading from CSV files or constant values."""

from pathlib import Path
from typing import Any

import numpy as np


class InflationCurve:
    """
    Inflation rate curve loaded from CSV or constant value.

    Provides year-indexed inflation rates for escalation calculations.
    Handles both single-value (constant) and CSV-based (time-varying) modes.

    Args:
        inflation_config: Either {'base_rate': float} or {'csv_path': str, ...}
        n_years: Project lifetime for array pre-allocation.

    Example (constant):
        >>> curve = InflationCurve({'base_rate': 0.02}, n_years=25)
        >>> curve.get_rate(0)  # Year 0
        0.02
        >>> curve.get_rate(24)  # Year 24
        0.02

    Example (CSV):
        >>> curve = InflationCurve({'csv_path': 'inflation.csv'}, n_years=25)
        >>> curve.get_rate(0)  # Year 0 from CSV
        0.02
        >>> curve.get_rates()  # All years as numpy array
        array([0.02, 0.025, 0.03, ...])
    """

    def __init__(
        self,
        inflation_config: dict[str, Any],
        n_years: int,
    ) -> None:
        """Initialize inflation curve from config."""
        self.n_years = n_years
        self._rates: np.ndarray | None = None

        if "base_rate" in inflation_config:
            # Constant rate mode
            rate = inflation_config["base_rate"]
            self._rates = np.full(n_years, rate, dtype=np.float64)
        elif "csv_path" in inflation_config:
            # CSV mode
            self._load_csv(inflation_config)
        else:
            raise ValueError(
                "Inflation config must contain 'base_rate' or 'csv_path'"
            )

    def _load_csv(self, config: dict[str, Any]) -> None:
        """Load inflation rates from CSV file."""
        csv_path = Path(config["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Inflation CSV not found: {csv_path}")

        rate_column = config.get("rate_column", "rate")

        # Try pandas first for better column handling, fall back to numpy
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)

            if rate_column not in df.columns:
                # Try common alternatives
                for alt in ["inflation_rate", "inflation", "rate"]:
                    if alt in df.columns:
                        rate_column = alt
                        break
                else:
                    raise ValueError(
                        f"Rate column '{rate_column}' not found in CSV. "
                        f"Available: {list(df.columns)}"
                    )

            loaded_rates = df[rate_column].values.astype(np.float64)
        except ImportError:
            # Numpy fallback - assume rate is in second column
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if data.ndim > 1:
                loaded_rates = data[:, 1].astype(np.float64)
            else:
                loaded_rates = data.astype(np.float64)

        # Validate and pad/truncate to project length
        self._rates = self._align_to_project(loaded_rates)

    def _align_to_project(self, loaded: np.ndarray) -> np.ndarray:
        """Align loaded rates to project lifetime."""
        if len(loaded) >= self.n_years:
            # Truncate to project length
            return loaded[: self.n_years].copy()
        else:
            # Extrapolate: use last value for remaining years
            result = np.zeros(self.n_years, dtype=np.float64)
            result[: len(loaded)] = loaded
            result[len(loaded) :] = loaded[-1]  # Carry forward last rate
            return result

    def get_rate(self, year: int) -> float:
        """
        Get inflation rate for a specific year.

        Args:
            year: Year index (0-based).

        Returns:
            Inflation rate as decimal (e.g., 0.02 for 2%).
        """
        if self._rates is None:
            raise RuntimeError("Inflation rates not loaded")

        if year < 0:
            year = 0
        elif year >= self.n_years:
            year = self.n_years - 1

        return float(self._rates[year])

    def get_rates(self) -> np.ndarray:
        """
        Get all inflation rates as numpy array.

        Returns:
            Array of shape (n_years,) with inflation rates.
        """
        if self._rates is None:
            raise RuntimeError("Inflation rates not loaded")
        return self._rates.copy()

    def get_cumulative_factors(self) -> np.ndarray:
        """
        Get cumulative escalation factors for compounding.

        Returns factors where:
        - Year 0: 1.0 (base year, no escalation)
        - Year 1: (1 + rate[0])
        - Year 2: (1 + rate[0]) * (1 + rate[1])
        - etc.

        Returns:
            Array of shape (n_years,) with cumulative factors.
        """
        if self._rates is None:
            raise RuntimeError("Inflation rates not loaded")

        # Calculate cumulative product of (1 + rate)
        cumulative = np.cumprod(1 + self._rates)
        # Shift so year 0 = 1.0, year 1 = (1+r0), etc.
        return np.insert(cumulative[:-1], 0, 1.0)

    @property
    def is_constant(self) -> bool:
        """Check if all rates are the same (constant mode)."""
        if self._rates is None:
            return False
        return bool(np.allclose(self._rates, self._rates[0]))
