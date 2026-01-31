"""Cost model for CAPEX and OPEX calculations."""

from typing import Any

import numpy as np


class CostModel:
    """
    Calculates capital and operating expenditures.

    Handles development, construction, replacement, and decommissioning CAPEX,
    as well as fixed and variable OPEX with escalation.
    """

    def calculate(
        self,
        capex_config: dict[str, Any],
        opex_config: dict[str, Any],
        inflation_rates: np.ndarray,
        n_years: int,
        n_months: int,
    ) -> dict[str, Any]:
        """
        Calculate all costs for the project.

        Args:
            capex_config: CAPEX configuration (development, construction, etc.).
            opex_config: OPEX configuration (fixed and variable).
            inflation_rates: Year-indexed inflation rates array of shape (n_years,).
            n_years: Project lifetime in years.
            n_months: Project lifetime in months.

        Returns:
            Dictionary with annual/monthly cost arrays and aggregates.
        """
        # Calculate CAPEX components
        capex_annual = self._calculate_capex(capex_config, n_years)

        # Calculate OPEX
        opex_fixed_annual = self._calculate_opex_fixed(
            opex_config.get("fixed", []), inflation_rates, n_years
        )
        opex_variable_monthly = self._calculate_opex_variable(
            opex_config.get("variable", []), inflation_rates, n_months
        )

        # Calculate totals
        total_depreciable = self._calculate_total_depreciable_capex(capex_config)
        initial_investment = self._calculate_initial_investment(capex_config)

        # Aggregate variable OPEX to annual
        opex_variable_annual = opex_variable_monthly.reshape(n_years, 12).sum(axis=1)

        # Total costs for LCOE
        total_costs_annual = capex_annual + opex_fixed_annual + opex_variable_annual

        # Handle decommissioning reserves if configured
        decom = capex_config.get("decommissioning", {})
        if decom.get("enabled", False) and decom.get("reserve_annual", False):
            annual_reserve = decom["amount"] / n_years
            opex_fixed_annual += annual_reserve

        return {
            "capex_annual": capex_annual,
            "opex_fixed_annual": opex_fixed_annual,
            "opex_variable_monthly": opex_variable_monthly,
            "opex_variable_annual": opex_variable_annual,
            "total_depreciable_capex": total_depreciable,
            "initial_investment": initial_investment,
            "total_costs_annual": total_costs_annual,
        }

    def _calculate_capex(
        self, capex_config: dict[str, Any], n_years: int
    ) -> np.ndarray:
        """
        Calculate annual CAPEX from all phases.

        Args:
            capex_config: CAPEX configuration.
            n_years: Project lifetime in years.

        Returns:
            Array of CAPEX by year.
        """
        capex = np.zeros(n_years)

        # Development costs
        for item in capex_config.get("development", []):
            year = item.get("year", 0)
            if year < 0:
                year = 0  # Map pre-construction to year 0
            if 0 <= year < n_years:
                capex[year] += item["amount"]

        # Construction costs
        for item in capex_config.get("construction", []):
            year = item.get("year", 0)
            if 0 <= year < n_years:
                capex[year] += item["amount"]

        # Replacement costs
        for item in capex_config.get("replacement", []):
            year = item.get("year", 0)
            if 0 <= year < n_years:
                capex[year] += item["amount"]

        # Decommissioning (one-time, if not reserve)
        decom = capex_config.get("decommissioning", {})
        if decom.get("enabled", False) and not decom.get("reserve_annual", False):
            decom_year = decom.get("year", n_years - 1)
            if 0 <= decom_year < n_years:
                capex[decom_year] += decom["amount"]

        return capex

    def _calculate_opex_fixed(
        self,
        fixed_config: list[dict[str, Any]],
        inflation_rates: np.ndarray,
        n_years: int,
    ) -> np.ndarray:
        """
        Calculate annual fixed OPEX with escalation.

        Supports both constant escalation rates (via escalation_rate parameter)
        and time-varying inflation (via inflation_rates array).

        Args:
            fixed_config: List of fixed OPEX items.
            inflation_rates: Year-indexed inflation rates array.
            n_years: Project lifetime.

        Returns:
            Array of fixed OPEX by year.
        """
        opex = np.zeros(n_years)

        # Pre-calculate cumulative factors for time-varying inflation
        # Year 0: 1.0, Year 1: (1+r0), Year 2: (1+r0)*(1+r1), etc.
        cumulative_factors = np.cumprod(1 + inflation_rates)
        cumulative_factors = np.insert(cumulative_factors[:-1], 0, 1.0)

        for item in fixed_config:
            annual_amount = item["annual_amount"]
            indexed = item.get("indexed", False)
            custom_escalation = item.get("escalation_rate")

            for year in range(n_years):
                if indexed:
                    if custom_escalation is not None:
                        # Use constant custom escalation rate
                        opex[year] += annual_amount * (1 + custom_escalation) ** year
                    else:
                        # Use time-varying inflation via cumulative factors
                        opex[year] += annual_amount * cumulative_factors[year]
                else:
                    opex[year] += annual_amount

        return opex

    def _calculate_opex_variable(
        self,
        variable_config: list[dict[str, Any]],
        inflation_rates: np.ndarray,
        n_months: int,
    ) -> np.ndarray:
        """
        Calculate monthly variable OPEX.

        Note: Volume drivers would be passed separately in full implementation.
        Currently returns zeros as placeholder.

        Args:
            variable_config: List of variable OPEX items.
            inflation_rates: Year-indexed inflation rates array.
            n_months: Project lifetime in months.

        Returns:
            Array of variable OPEX by month.
        """
        # Variable OPEX depends on volume drivers
        # In full implementation, this would multiply unit_cost by volume
        return np.zeros(n_months)

    def _calculate_total_depreciable_capex(
        self, capex_config: dict[str, Any]
    ) -> float:
        """
        Calculate total CAPEX that can be depreciated.

        Includes development, construction, and replacement costs.
        Typically excludes land.

        Args:
            capex_config: CAPEX configuration.

        Returns:
            Total depreciable CAPEX amount.
        """
        total = 0.0

        for item in capex_config.get("development", []):
            total += item["amount"]

        for item in capex_config.get("construction", []):
            total += item["amount"]

        for item in capex_config.get("replacement", []):
            total += item["amount"]

        return total

    def _calculate_initial_investment(self, capex_config: dict[str, Any]) -> float:
        """
        Calculate initial investment (development + construction).

        Args:
            capex_config: CAPEX configuration.

        Returns:
            Initial investment amount.
        """
        total = 0.0

        for item in capex_config.get("development", []):
            total += item["amount"]

        for item in capex_config.get("construction", []):
            total += item["amount"]

        return total
