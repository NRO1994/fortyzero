"""KPI calculator for DSCR, Payback Period, and LCOE."""

from typing import Any

import numpy as np

from financial_model.core.dcf_engine import DCFEngine


class KPICalculator:
    """
    Calculates financial KPIs: DSCR, Payback Period, LCOE.

    Provides comprehensive financial metrics for project evaluation.
    """

    def __init__(self) -> None:
        """Initialize KPI calculator."""
        self.dcf_engine = DCFEngine()

    def calculate(
        self,
        fcf_unlevered: np.ndarray,
        fcf_levered: np.ndarray,
        initial_investment: float,
        equity_amount: float,
        wacc: float,
        cost_of_equity: float | None,
        annual_ebitda: np.ndarray,
        annual_tax: np.ndarray,
        annual_capex: np.ndarray,
        debt_service: np.ndarray,
        total_costs: np.ndarray,
        interest_payments: np.ndarray,
        annual_volumes: np.ndarray,
        n_years: int,
    ) -> dict[str, Any]:
        """
        Calculate all financial KPIs.

        Args:
            fcf_unlevered: Unlevered free cash flows.
            fcf_levered: Levered free cash flows.
            initial_investment: Total initial CAPEX.
            equity_amount: Equity invested.
            wacc: Weighted average cost of capital.
            cost_of_equity: Cost of equity (for equity IRR discount).
            annual_ebitda: EBITDA by year.
            annual_tax: Tax by year.
            annual_capex: CAPEX by year.
            debt_service: Total debt service (interest + principal) by year.
            total_costs: Total costs by year for LCOE.
            interest_payments: Interest by year.
            annual_volumes: Energy production by year.
            n_years: Project lifetime.

        Returns:
            Dictionary with all KPIs including NPV, IRR, DSCR, Payback, LCOE.
        """
        # NPV calculations
        npv_project = self.dcf_engine.calculate_npv(fcf_unlevered, wacc, n_years)
        discount_rate_equity = cost_of_equity if cost_of_equity else wacc
        npv_equity = self.dcf_engine.calculate_npv(
            fcf_levered, discount_rate_equity, n_years
        )

        # IRR calculations
        # For project IRR: include initial investment as negative at t=0
        cash_flows_project = fcf_unlevered.copy()
        cash_flows_project[0] = -initial_investment + fcf_unlevered[0]
        irr_project = self.dcf_engine.calculate_irr(cash_flows_project)

        # For equity IRR: fcf_levered already includes equity investment at t=0
        irr_equity = self.dcf_engine.calculate_irr(fcf_levered)

        # DSCR calculation
        dscr_series = self._calculate_dscr(
            annual_ebitda, annual_tax, annual_capex, debt_service
        )

        # Payback calculations
        payback_simple = self._calculate_payback_simple(fcf_levered)
        payback_discounted = self._calculate_payback_discounted(fcf_levered, wacc)

        # LCOE calculation
        lcoe = self._calculate_lcoe(
            total_costs, interest_payments, annual_tax, annual_volumes, wacc, n_years
        )

        return {
            "npv_project": npv_project,
            "npv_equity": npv_equity,
            "irr_project": irr_project,
            "irr_equity": irr_equity,
            "payback_simple": payback_simple,
            "payback_discounted": payback_discounted,
            "dscr_min": float(np.min(dscr_series[dscr_series > 0]))
            if np.any(dscr_series > 0)
            else float("inf"),
            "dscr_avg": float(np.mean(dscr_series[dscr_series > 0]))
            if np.any(dscr_series > 0)
            else float("inf"),
            "dscr_series": dscr_series,
            "lcoe": lcoe,
        }

    @staticmethod
    def _calculate_dscr(
        ebitda: np.ndarray,
        tax: np.ndarray,
        capex: np.ndarray,
        debt_service: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Debt Service Coverage Ratio by year.

        DSCR = Cash Available for Debt Service / Debt Service

        Args:
            ebitda: EBITDA by year.
            tax: Tax by year.
            capex: CAPEX by year.
            debt_service: Total debt service by year.

        Returns:
            Array of DSCR values by year.
        """
        cash_available = ebitda - tax - capex
        # Avoid division by zero
        dscr = np.where(
            debt_service > 0, cash_available / debt_service, np.inf
        )
        return dscr

    @staticmethod
    def _calculate_payback_simple(cash_flows: np.ndarray) -> float:
        """
        Calculate simple (undiscounted) payback period.

        Args:
            cash_flows: Annual cash flows.

        Returns:
            Payback period in years. np.inf if never recovered.
        """
        cumulative = np.cumsum(cash_flows)
        positive_indices = np.where(cumulative > 0)[0]

        if len(positive_indices) == 0:
            return float("inf")

        # First year with positive cumulative
        payback_year = positive_indices[0]

        # Interpolate for fractional year
        if payback_year > 0:
            prev_cumulative = cumulative[payback_year - 1]
            year_cf = cash_flows[payback_year]
            if year_cf != 0:
                fraction = -prev_cumulative / year_cf
                return float(payback_year - 1 + fraction)

        return float(payback_year)

    @staticmethod
    def _calculate_payback_discounted(
        cash_flows: np.ndarray, discount_rate: float
    ) -> float:
        """
        Calculate discounted payback period.

        Args:
            cash_flows: Annual cash flows.
            discount_rate: Discount rate.

        Returns:
            Discounted payback period in years. np.inf if never recovered.
        """
        n_years = len(cash_flows)
        years = np.arange(n_years)
        discount_factors = 1 / (1 + discount_rate) ** years
        discount_factors[0] = 1.0

        discounted_cf = cash_flows * discount_factors
        cumulative = np.cumsum(discounted_cf)
        positive_indices = np.where(cumulative > 0)[0]

        if len(positive_indices) == 0:
            return float("inf")

        payback_year = positive_indices[0]

        if payback_year > 0:
            prev_cumulative = cumulative[payback_year - 1]
            year_dcf = discounted_cf[payback_year]
            if year_dcf != 0:
                fraction = -prev_cumulative / year_dcf
                return float(payback_year - 1 + fraction)

        return float(payback_year)

    @staticmethod
    def _calculate_lcoe(
        total_costs: np.ndarray,
        interest: np.ndarray,
        tax: np.ndarray,
        volumes: np.ndarray,
        wacc: float,
        n_years: int,
    ) -> float:
        """
        Calculate Levelized Cost of Energy.

        LCOE = NPV(Total Costs) / NPV(Total Energy Production)

        Args:
            total_costs: Total costs by year (CAPEX + OPEX).
            interest: Interest payments by year.
            tax: Tax by year.
            volumes: Energy production by year.
            wacc: Discount rate.
            n_years: Project lifetime.

        Returns:
            LCOE in currency per energy unit.
        """
        years = np.arange(n_years)
        discount_factors = 1 / (1 + wacc) ** years
        discount_factors[0] = 1.0

        all_costs = total_costs + interest + tax
        npv_costs = float(np.sum(all_costs * discount_factors))
        npv_energy = float(np.sum(volumes * discount_factors))

        if npv_energy == 0:
            return float("inf")

        return npv_costs / npv_energy
