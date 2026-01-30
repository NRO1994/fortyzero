"""Financing model for debt service calculations."""

from typing import Any

import numpy as np

from financial_model.utils.financial_utils import calculate_annuity_payment


class FinancingModel:
    """
    Calculates debt service for annuity loans.

    Handles interest and principal payments with constant annuity structure.
    """

    def calculate(
        self,
        financing_config: dict[str, Any],
        n_years: int,
    ) -> dict[str, Any]:
        """
        Calculate debt service over the loan term.

        Args:
            financing_config: Financing configuration with equity_share and debt.
            n_years: Project lifetime.

        Returns:
            Dictionary with debt balance, interest, principal, and totals.
        """
        equity_share = financing_config.get("equity_share", 1.0)
        debt_config = financing_config.get("debt", {})

        # If no debt, return zeros
        if equity_share >= 1.0 or not debt_config:
            return self._no_debt_results(n_years)

        principal = debt_config["principal"]
        interest_rate = debt_config["interest_rate"]
        term_years = debt_config["term_years"]

        # Calculate equity amount (based on total project cost)
        # Assuming principal represents debt portion
        total_cost = principal / (1 - equity_share)
        equity_amount = total_cost * equity_share

        # Calculate annuity payment
        annuity = calculate_annuity_payment(principal, interest_rate, term_years)

        # Calculate payment schedule
        debt_balance = np.zeros(n_years + 1)
        interest_payments = np.zeros(n_years)
        principal_payments = np.zeros(n_years)

        debt_balance[0] = principal

        for year in range(min(term_years, n_years)):
            interest_payments[year] = debt_balance[year] * interest_rate
            principal_payments[year] = annuity - interest_payments[year]
            debt_balance[year + 1] = debt_balance[year] - principal_payments[year]

            # Handle floating point precision
            if debt_balance[year + 1] < 0.01:
                debt_balance[year + 1] = 0.0

        debt_service = interest_payments + principal_payments

        return {
            "equity_amount": equity_amount,
            "debt_drawdown": principal,
            "debt_balance": debt_balance[:-1],  # Exclude final zero
            "interest_payments": interest_payments,
            "principal_payments": principal_payments,
            "debt_service": debt_service,
            "annuity": annuity,
        }

    @staticmethod
    def _no_debt_results(n_years: int) -> dict[str, Any]:
        """
        Return zero-filled results for 100% equity financing.

        Args:
            n_years: Project lifetime.

        Returns:
            Dictionary with zeros for all debt-related values.
        """
        return {
            "equity_amount": 0.0,  # Will be set by caller based on CAPEX
            "debt_drawdown": 0.0,
            "debt_balance": np.zeros(n_years),
            "interest_payments": np.zeros(n_years),
            "principal_payments": np.zeros(n_years),
            "debt_service": np.zeros(n_years),
            "annuity": 0.0,
        }
