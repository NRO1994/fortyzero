"""Main FinancialModel class orchestrating all calculations."""

from typing import Any

import numpy as np

from financial_model.core.dcf_engine import DCFEngine
from financial_model.core.kpi_calculator import KPICalculator
from financial_model.interfaces.volume_model import VolumeModelInterface
from financial_model.interfaces.price_curve import PriceCurveInterface
from financial_model.models.revenue_model import RevenueModel
from financial_model.models.cost_model import CostModel
from financial_model.models.tax_model import TaxModel
from financial_model.models.financing_model import FinancingModel
from financial_model.models.inflation_curve import InflationCurve
from financial_model.templates.asset_templates import ASSET_TEMPLATES


class FinancialModel:
    """
    Main orchestrator for financial calculations.

    Coordinates all sub-models and calculates comprehensive financial KPIs
    for energy infrastructure projects. Supports both hourly and monthly
    granularity for revenue calculations.

    Args:
        asset_type: Type of asset ('pv', 'wind', 'heat_network', 'chp').
        volume_model: Optional VolumeModel instance for production calculation.
        price_curve: Optional PriceCurve instance for market prices.

    Example:
        >>> from financial_model.models import PVVolumeModel, CSVPriceCurve
        >>> volume_model = PVVolumeModel()
        >>> price_curve = CSVPriceCurve(scenario='mid', csv_path='prices.csv')
        >>> model = FinancialModel(
        ...     asset_type='pv',
        ...     volume_model=volume_model,
        ...     price_curve=price_curve
        ... )
        >>> results = model.calculate_hourly(params)
        >>> print(results['kpis']['npv_project'])
    """

    HOURS_PER_YEAR = 8760

    def __init__(
        self,
        asset_type: str,
        volume_model: VolumeModelInterface | None = None,
        price_curve: PriceCurveInterface | None = None,
    ) -> None:
        """Initialize the financial model with asset type and optional interfaces."""
        if asset_type not in ASSET_TEMPLATES:
            raise ValueError(
                f"Unknown asset_type '{asset_type}'. "
                f"Available types: {list(ASSET_TEMPLATES.keys())}"
            )

        self.asset_type = asset_type
        self.template = ASSET_TEMPLATES[asset_type]
        self.volume_model = volume_model
        self.price_curve = price_curve

        # Initialize sub-models
        self.revenue_model = RevenueModel()
        self.cost_model = CostModel()
        self.tax_model = TaxModel()
        self.financing_model = FinancingModel()
        self.dcf_engine = DCFEngine()
        self.kpi_calculator = KPICalculator()

    def calculate_hourly(
        self,
        params: dict[str, Any],
        hourly_volumes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Calculate all financial KPIs using hourly granularity.

        Uses hourly volumes and hourly prices for accurate revenue calculation,
        especially important for market price periods with variable pricing.

        Args:
            params: Full parameter dictionary as specified in the spec.
            hourly_volumes: Pre-calculated hourly production volumes.
                Shape: (n_years × 8760,). If None, uses volume_model.

        Returns:
            Dictionary containing:
                - kpis: Primary KPIs (NPV, IRR, DSCR, Payback, LCOE)
                - cash_flows: Annual, monthly, and hourly cash flows
                - financing: Debt balance and DSCR time series

        Raises:
            ValueError: If hourly_volumes is None and no volume_model provided.
        """
        # Extract project parameters
        project = params["project"]
        financial = params["financial"]
        n_years = project["lifetime_years"]
        n_hours = n_years * self.HOURS_PER_YEAR
        n_months = n_years * 12

        # Get or calculate hourly volumes
        if hourly_volumes is None:
            if self.volume_model is None:
                raise ValueError(
                    "Either hourly_volumes or volume_model must be provided"
                )
            hourly_volumes = self.volume_model.calculate(params["technical"])

        # Validate shape
        if len(hourly_volumes) != n_hours:
            raise ValueError(
                f"Expected {n_hours} hourly volumes ({n_years} years × 8760), "
                f"got {len(hourly_volumes)}"
            )

        # Apply template defaults
        tax_params = self._apply_template_defaults(financial.get("tax", {}))
        discount_params = financial.get("discount", {})

        # Load inflation rates (supports both constant and CSV-based)
        inflation_curve = InflationCurve(financial["inflation"], n_years)
        inflation_rates = inflation_curve.get_rates()

        # Calculate hourly revenues
        hourly_revenue = self.revenue_model.calculate_hourly(
            hourly_volumes=hourly_volumes,
            revenue_config=financial["revenue"],
            inflation_rates=inflation_rates,
            price_curve=self.price_curve,
            n_years=n_years,
        )

        # Aggregate to annual
        annual_revenue = self._aggregate_hourly_to_annual(hourly_revenue, n_years)
        annual_volumes = self._aggregate_hourly_to_annual(hourly_volumes, n_years)

        # Calculate costs (unchanged - costs are annual/monthly)
        cost_results = self.cost_model.calculate(
            capex_config=financial["capex"],
            opex_config=financial["opex"],
            inflation_rates=inflation_rates,
            n_years=n_years,
            n_months=n_months,
        )

        annual_opex_var = self._aggregate_monthly_to_annual(
            cost_results["opex_variable_monthly"], n_years
        )

        # Calculate financing
        financing_results = self.financing_model.calculate(
            financing_config=financial["financing"],
            n_years=n_years,
        )

        # Calculate taxes
        tax_results = self.tax_model.calculate(
            annual_revenue=annual_revenue,
            annual_opex_fixed=cost_results["opex_fixed_annual"],
            annual_opex_variable=annual_opex_var,
            annual_capex=cost_results["capex_annual"],
            interest_payments=financing_results["interest_payments"],
            tax_rate=tax_params["corporate_tax_rate"],
            depreciation_years=tax_params["depreciation_years"],
            total_depreciable_capex=cost_results["total_depreciable_capex"],
            n_years=n_years,
        )

        # Calculate cash flows via DCF engine
        dcf_results = self.dcf_engine.calculate(
            annual_revenue=annual_revenue,
            annual_opex_fixed=cost_results["opex_fixed_annual"],
            annual_opex_variable=annual_opex_var,
            annual_capex=cost_results["capex_annual"],
            annual_tax=tax_results["tax"],
            interest_payments=financing_results["interest_payments"],
            principal_payments=financing_results["principal_payments"],
            equity_amount=financing_results["equity_amount"],
            debt_drawdown=financing_results["debt_drawdown"],
            wacc=discount_params.get("wacc", self.template["typical_wacc"]),
            cost_of_equity=discount_params.get("cost_of_equity"),
            n_years=n_years,
        )

        # Calculate KPIs
        kpis = self.kpi_calculator.calculate(
            fcf_unlevered=dcf_results["fcf_unlevered"],
            fcf_levered=dcf_results["fcf_levered"],
            initial_investment=cost_results["initial_investment"],
            equity_amount=financing_results["equity_amount"],
            wacc=discount_params.get("wacc", self.template["typical_wacc"]),
            cost_of_equity=discount_params.get("cost_of_equity"),
            annual_ebitda=dcf_results["ebitda"],
            annual_tax=tax_results["tax"],
            annual_capex=cost_results["capex_annual"],
            debt_service=financing_results["debt_service"],
            total_costs=cost_results["total_costs_annual"],
            interest_payments=financing_results["interest_payments"],
            annual_volumes=annual_volumes,
            n_years=n_years,
        )

        # Aggregate hourly to monthly for output
        monthly_revenue = RevenueModel.aggregate_hourly_to_monthly(hourly_revenue)
        monthly_volumes = self._aggregate_hourly_to_monthly(hourly_volumes)

        return {
            "kpis": kpis,
            "cash_flows": {
                "annual": {
                    "revenue": annual_revenue,
                    "opex_fixed": cost_results["opex_fixed_annual"],
                    "opex_variable": annual_opex_var,
                    "capex": cost_results["capex_annual"],
                    "depreciation": tax_results["depreciation"],
                    "tax": tax_results["tax"],
                    "interest": financing_results["interest_payments"],
                    "principal": financing_results["principal_payments"],
                    "fcf_unlevered": dcf_results["fcf_unlevered"],
                    "fcf_levered": dcf_results["fcf_levered"],
                },
                "monthly": {
                    "revenue": monthly_revenue,
                    "opex_variable": cost_results["opex_variable_monthly"],
                    "volume": monthly_volumes,
                },
                "hourly": {
                    "revenue": hourly_revenue,
                    "volume": hourly_volumes,
                },
            },
            "financing": {
                "debt_balance": financing_results["debt_balance"],
                "dscr": kpis["dscr_series"],
            },
        }

    def calculate(
        self,
        params: dict[str, Any],
        monthly_volumes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Calculate all financial KPIs using monthly granularity.

        Backward-compatible method that works with monthly volumes.
        For more accurate results with variable market prices, use
        calculate_hourly() instead.

        Args:
            params: Full parameter dictionary as specified in the spec.
            monthly_volumes: Pre-calculated monthly production volumes.
                If None, uses volume_model to calculate.

        Returns:
            Dictionary containing:
                - kpis: Primary KPIs (NPV, IRR, DSCR, Payback, LCOE)
                - cash_flows: Detailed annual and monthly cash flows
                - financing: Debt balance and DSCR time series

        Raises:
            ValueError: If monthly_volumes is None and no volume_model provided.
        """
        # Extract project parameters
        project = params["project"]
        financial = params["financial"]
        n_years = project["lifetime_years"]
        n_months = n_years * 12

        # Get or calculate monthly volumes
        if monthly_volumes is None:
            if self.volume_model is None:
                raise ValueError(
                    "Either monthly_volumes or volume_model must be provided"
                )
            # If volume_model returns hourly, convert to monthly
            hourly_volumes = self.volume_model.calculate(params["technical"])
            if len(hourly_volumes) == n_years * self.HOURS_PER_YEAR:
                monthly_volumes = self._aggregate_hourly_to_monthly(hourly_volumes)
            else:
                monthly_volumes = hourly_volumes  # Assume already monthly

        # Apply template defaults where not overridden
        tax_params = self._apply_template_defaults(financial.get("tax", {}))
        discount_params = financial.get("discount", {})

        # Load inflation rates (supports both constant and CSV-based)
        inflation_curve = InflationCurve(financial["inflation"], n_years)
        inflation_rates = inflation_curve.get_rates()

        # Calculate revenues (monthly)
        monthly_revenue = self.revenue_model.calculate(
            volumes=monthly_volumes,
            revenue_config=financial["revenue"],
            inflation_rates=inflation_rates,
            price_curve=self.price_curve,
            n_months=n_months,
        )

        # Calculate costs
        cost_results = self.cost_model.calculate(
            capex_config=financial["capex"],
            opex_config=financial["opex"],
            inflation_rates=inflation_rates,
            n_years=n_years,
            n_months=n_months,
        )

        # Aggregate monthly to annual
        annual_revenue = self._aggregate_monthly_to_annual(monthly_revenue, n_years)
        annual_opex_var = self._aggregate_monthly_to_annual(
            cost_results["opex_variable_monthly"], n_years
        )

        # Calculate financing
        financing_results = self.financing_model.calculate(
            financing_config=financial["financing"],
            n_years=n_years,
        )

        # Calculate taxes
        tax_results = self.tax_model.calculate(
            annual_revenue=annual_revenue,
            annual_opex_fixed=cost_results["opex_fixed_annual"],
            annual_opex_variable=annual_opex_var,
            annual_capex=cost_results["capex_annual"],
            interest_payments=financing_results["interest_payments"],
            tax_rate=tax_params["corporate_tax_rate"],
            depreciation_years=tax_params["depreciation_years"],
            total_depreciable_capex=cost_results["total_depreciable_capex"],
            n_years=n_years,
        )

        # Calculate cash flows via DCF engine
        dcf_results = self.dcf_engine.calculate(
            annual_revenue=annual_revenue,
            annual_opex_fixed=cost_results["opex_fixed_annual"],
            annual_opex_variable=annual_opex_var,
            annual_capex=cost_results["capex_annual"],
            annual_tax=tax_results["tax"],
            interest_payments=financing_results["interest_payments"],
            principal_payments=financing_results["principal_payments"],
            equity_amount=financing_results["equity_amount"],
            debt_drawdown=financing_results["debt_drawdown"],
            wacc=discount_params.get("wacc", self.template["typical_wacc"]),
            cost_of_equity=discount_params.get("cost_of_equity"),
            n_years=n_years,
        )

        # Calculate KPIs
        kpis = self.kpi_calculator.calculate(
            fcf_unlevered=dcf_results["fcf_unlevered"],
            fcf_levered=dcf_results["fcf_levered"],
            initial_investment=cost_results["initial_investment"],
            equity_amount=financing_results["equity_amount"],
            wacc=discount_params.get("wacc", self.template["typical_wacc"]),
            cost_of_equity=discount_params.get("cost_of_equity"),
            annual_ebitda=dcf_results["ebitda"],
            annual_tax=tax_results["tax"],
            annual_capex=cost_results["capex_annual"],
            debt_service=financing_results["debt_service"],
            total_costs=cost_results["total_costs_annual"],
            interest_payments=financing_results["interest_payments"],
            annual_volumes=self._aggregate_monthly_to_annual(monthly_volumes, n_years),
            n_years=n_years,
        )

        return {
            "kpis": kpis,
            "cash_flows": {
                "annual": {
                    "revenue": annual_revenue,
                    "opex_fixed": cost_results["opex_fixed_annual"],
                    "opex_variable": annual_opex_var,
                    "capex": cost_results["capex_annual"],
                    "depreciation": tax_results["depreciation"],
                    "tax": tax_results["tax"],
                    "interest": financing_results["interest_payments"],
                    "principal": financing_results["principal_payments"],
                    "fcf_unlevered": dcf_results["fcf_unlevered"],
                    "fcf_levered": dcf_results["fcf_levered"],
                },
                "monthly": {
                    "revenue": monthly_revenue,
                    "opex_variable": cost_results["opex_variable_monthly"],
                    "volume": monthly_volumes,
                },
            },
            "financing": {
                "debt_balance": financing_results["debt_balance"],
                "dscr": kpis["dscr_series"],
            },
        }

    def _apply_template_defaults(self, tax_params: dict[str, Any]) -> dict[str, Any]:
        """Apply asset template defaults for missing tax parameters."""
        return {
            "corporate_tax_rate": tax_params.get(
                "corporate_tax_rate", self.template["corporate_tax_rate"]
            ),
            "depreciation_years": tax_params.get(
                "depreciation_years", self.template["depreciation_years"]
            ),
            "depreciation_method": tax_params.get("depreciation_method", "linear"),
        }

    @staticmethod
    def _aggregate_monthly_to_annual(
        monthly_values: np.ndarray, n_years: int
    ) -> np.ndarray:
        """Aggregate monthly values to annual totals."""
        return monthly_values.reshape(n_years, 12).sum(axis=1)

    @staticmethod
    def _aggregate_hourly_to_annual(
        hourly_values: np.ndarray, n_years: int
    ) -> np.ndarray:
        """Aggregate hourly values to annual totals."""
        hours_per_year = 8760
        return hourly_values.reshape(n_years, hours_per_year).sum(axis=1)

    @staticmethod
    def _aggregate_hourly_to_monthly(hourly_values: np.ndarray) -> np.ndarray:
        """Aggregate hourly values to monthly totals (approximate 730h/month)."""
        hours_per_month = 730
        n_months = len(hourly_values) // hours_per_month
        monthly = np.zeros(n_months)

        for month in range(n_months):
            start = month * hours_per_month
            end = start + hours_per_month
            monthly[month] = np.sum(hourly_values[start:end])

        # Handle remaining hours
        remaining = len(hourly_values) - (n_months * hours_per_month)
        if remaining > 0 and n_months > 0:
            monthly[-1] += np.sum(hourly_values[-remaining:])

        return monthly
