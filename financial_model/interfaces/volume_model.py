"""Abstract interface for volume/energy production models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VolumeModelInterface(ABC):
    """
    Abstract interface for volume/energy production models.

    Implementations handle asset-specific physics, degradation, and availability.
    The VolumeModel is NOT part of the core FinancialModel - it is developed
    separately and passed via dependency injection.

    Example implementations:
        - PVVolumeModel: Solar irradiance × capacity × degradation × availability
        - WindVolumeModel: Wind resource × capacity × power curve × availability
        - HeatNetworkVolumeModel: Customer growth × consumption × seasonality
    """

    @abstractmethod
    def calculate(self, params: dict[str, Any]) -> np.ndarray:
        """
        Calculate monthly production volumes.

        Args:
            params: Technical parameters from input dict.
                Expected contents vary by implementation but typically include:
                - capacity: Installed capacity (kW, MW)
                - degradation_rate: Annual degradation rate
                - availability: System availability factor

        Returns:
            np.ndarray of shape (n_months,) with production in kWh, MWh,
            or other relevant units.

        Example for PV:
            >>> class PVVolumeModel(VolumeModelInterface):
            ...     def calculate(self, params):
            ...         capacity = params['capacity']  # kW
            ...         irradiance = params['irradiance_profile']  # kWh/kW/month
            ...         degradation = params['degradation_rate']  # e.g., 0.005
            ...         availability = params['availability']  # e.g., 0.97
            ...
            ...         n_months = len(irradiance)
            ...         production = np.zeros(n_months)
            ...
            ...         for month in range(n_months):
            ...             year = month // 12
            ...             deg_factor = (1 - degradation) ** year
            ...             production[month] = (
            ...                 capacity * irradiance[month] * deg_factor * availability
            ...             )
            ...
            ...         return production

        Example for Heat Network:
            >>> class HeatNetworkVolumeModel(VolumeModelInterface):
            ...     def calculate(self, params):
            ...         customers = params['customer_growth_curve']  # per month
            ...         consumption = params['consumption_per_customer']  # MWh/month
            ...         seasonality = params['seasonality_factors']  # 12 values
            ...
            ...         n_months = len(customers)
            ...         production = np.zeros(n_months)
            ...
            ...         for month in range(n_months):
            ...             season_idx = month % 12
            ...             production[month] = (
            ...                 customers[month] * consumption * seasonality[season_idx]
            ...             )
            ...
            ...         return production
        """
        raise NotImplementedError
