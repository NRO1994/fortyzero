"""Discounted Cash Flow engine for NPV and IRR calculations."""

from typing import Any

import numpy as np
from scipy import optimize


class DCFEngine:
    """
    Calculates discounted cash flows, NPV, and IRR.

    Handles both unlevered (project) and levered (equity) cash flows
    with end-of-period discounting convention.
    """

    def calculate(
        self,
        annual_revenue: np.ndarray,
        annual_opex_fixed: np.ndarray,
        annual_opex_variable: np.ndarray,
        annual_capex: np.ndarray,
        annual_tax: np.ndarray,
        interest_payments: np.ndarray,
        principal_payments: np.ndarray,
        equity_amount: float,
        debt_drawdown: float,
        wacc: float,
        cost_of_equity: float | None,
        n_years: int,
    ) -> dict[str, Any]:
        """
        Calculate cash flows and core DCF metrics.

        Args:
            annual_revenue: Revenue by year.
            annual_opex_fixed: Fixed operating costs by year.
            annual_opex_variable: Variable operating costs by year.
            annual_capex: Capital expenditures by year.
            annual_tax: Tax liability by year.
            interest_payments: Interest payments by year.
            principal_payments: Principal payments by year.
            equity_amount: Total equity invested.
            debt_drawdown: Total debt drawn at start.
            wacc: Weighted average cost of capital.
            cost_of_equity: Cost of equity for levered NPV.
            n_years: Project lifetime in years.

        Returns:
            Dictionary with EBITDA, unlevered FCF, levered FCF, and NPVs.
        """
        # Calculate EBITDA
        ebitda = annual_revenue - annual_opex_fixed - annual_opex_variable

        # Unlevered Free Cash Flow (Project perspective)
        # Tax on EBIT (without interest deduction) for unlevered
        fcf_unlevered = ebitda - annual_capex - annual_tax

        # Levered Free Cash Flow (Equity perspective)
        fcf_levered = (
            ebitda
            - annual_capex
            - annual_tax
            - interest_payments
            - principal_payments
        )

        # Adjust year 0 for initial equity and debt
        fcf_levered[0] = fcf_levered[0] - equity_amount + debt_drawdown

        # Calculate NPVs
        npv_project = self.calculate_npv(fcf_unlevered, wacc, n_years)

        discount_rate_equity = cost_of_equity if cost_of_equity else wacc
        npv_equity = self.calculate_npv(fcf_levered, discount_rate_equity, n_years)

        return {
            "ebitda": ebitda,
            "fcf_unlevered": fcf_unlevered,
            "fcf_levered": fcf_levered,
            "npv_project": npv_project,
            "npv_equity": npv_equity,
        }

    @staticmethod
    def calculate_npv(
        cash_flows: np.ndarray,
        discount_rate: float,
        n_years: int,
    ) -> float:
        """
        Calculate Net Present Value with end-of-period convention.

        Args:
            cash_flows: Annual cash flows (year 0 to n_years-1).
            discount_rate: Discount rate (e.g., WACC).
            n_years: Number of years.

        Returns:
            NPV as float.
        """
        # Year indices: 0, 1, 2, ..., n_years-1
        # Discount factors: 1/(1+r)^1, 1/(1+r)^2, etc. for years 1+
        # Year 0 is not discounted (or discounted by factor 1)
        years = np.arange(n_years)
        discount_factors = 1 / (1 + discount_rate) ** years
        # Year 0 cash flow at t=0 (not discounted)
        discount_factors[0] = 1.0

        return float(np.sum(cash_flows * discount_factors))

    @staticmethod
    def calculate_irr(cash_flows: np.ndarray) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson.

        Args:
            cash_flows: Cash flow series with initial investment (negative) at index 0.

        Returns:
            IRR as decimal (e.g., 0.08 for 8%). Returns np.nan if no convergence.
        """

        def npv_func(rate: float) -> float:
            if rate <= -1:
                return float("inf")
            years = np.arange(len(cash_flows))
            return float(np.sum(cash_flows / (1 + rate) ** years))

        try:
            irr = optimize.newton(npv_func, x0=0.1, maxiter=100)
            return float(irr)
        except (RuntimeError, ValueError):
            # Try brentq as fallback
            try:
                irr = optimize.brentq(npv_func, -0.99, 10.0)
                return float(irr)
            except ValueError:
                return float("nan")
