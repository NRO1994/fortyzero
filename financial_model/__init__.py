"""
Financial Model for Energy Infrastructure Investment Analysis.

A robust, fast (<10ms per evaluation), and versatile financial model that:
- Calculates NPV, IRR (project & equity), Payback Period, DSCR, and LCOE
- Supports multiple energy asset types through modular architecture
- Integrates with Monte Carlo simulation framework
- Handles monthly operational granularity with annual DCF aggregation
"""

from financial_model.core.financial_model import FinancialModel
from financial_model.templates.asset_templates import ASSET_TEMPLATES

__version__ = "1.0.0"
__all__ = ["FinancialModel", "ASSET_TEMPLATES"]
