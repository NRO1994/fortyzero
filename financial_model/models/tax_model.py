"""Tax model for calculating depreciation and tax liability."""

from typing import Any

import numpy as np


class TaxModel:
    """
    Calculates tax liability with linear depreciation and loss carry-forward.

    Supports German corporate tax rules with infinite loss carry-forward.
    """

    def calculate(
        self,
        annual_revenue: np.ndarray,
        annual_opex_fixed: np.ndarray,
        annual_opex_variable: np.ndarray,
        annual_capex: np.ndarray,
        interest_payments: np.ndarray,
        tax_rate: float,
        depreciation_years: int,
        total_depreciable_capex: float,
        n_years: int,
    ) -> dict[str, Any]:
        """
        Calculate annual tax liability.

        Args:
            annual_revenue: Revenue by year.
            annual_opex_fixed: Fixed OPEX by year.
            annual_opex_variable: Variable OPEX by year.
            annual_capex: CAPEX by year (for timing, not depreciation base).
            interest_payments: Interest expense by year.
            tax_rate: Corporate tax rate (e.g., 0.30 for 30%).
            depreciation_years: Years over which to depreciate CAPEX.
            total_depreciable_capex: Total CAPEX to depreciate.
            n_years: Project lifetime.

        Returns:
            Dictionary with depreciation, taxable income, and tax arrays.
        """
        # Calculate linear depreciation
        depreciation = self._calculate_depreciation(
            total_depreciable_capex, depreciation_years, n_years
        )

        # Calculate EBITDA, EBIT, EBT
        ebitda = annual_revenue - annual_opex_fixed - annual_opex_variable
        ebit = ebitda - depreciation
        ebt = ebit - interest_payments

        # Calculate tax with loss carry-forward
        tax, accumulated_losses = self._calculate_tax_with_losses(
            ebt, tax_rate, n_years
        )

        return {
            "depreciation": depreciation,
            "ebitda": ebitda,
            "ebit": ebit,
            "ebt": ebt,
            "tax": tax,
            "accumulated_losses": accumulated_losses,
        }

    @staticmethod
    def _calculate_depreciation(
        total_capex: float,
        depreciation_years: int,
        n_years: int,
    ) -> np.ndarray:
        """
        Calculate linear depreciation schedule.

        Args:
            total_capex: Total amount to depreciate.
            depreciation_years: Depreciation period.
            n_years: Project lifetime.

        Returns:
            Array of annual depreciation amounts.
        """
        depreciation = np.zeros(n_years)
        annual_depreciation = total_capex / depreciation_years

        for year in range(min(depreciation_years, n_years)):
            depreciation[year] = annual_depreciation

        return depreciation

    @staticmethod
    def _calculate_tax_with_losses(
        ebt: np.ndarray,
        tax_rate: float,
        n_years: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate tax with infinite loss carry-forward.

        Args:
            ebt: Earnings before tax by year.
            tax_rate: Corporate tax rate.
            n_years: Project lifetime.

        Returns:
            Tuple of (tax array, accumulated losses array).
        """
        tax = np.zeros(n_years)
        accumulated_losses = np.zeros(n_years)
        loss_pool = 0.0

        for year in range(n_years):
            if ebt[year] < 0:
                # Add to loss pool
                loss_pool += abs(ebt[year])
                tax[year] = 0.0
            else:
                # Offset profits with accumulated losses
                taxable_income = max(0, ebt[year] - loss_pool)
                offset = min(ebt[year], loss_pool)
                loss_pool -= offset
                tax[year] = taxable_income * tax_rate

            accumulated_losses[year] = loss_pool

        return tax, accumulated_losses
