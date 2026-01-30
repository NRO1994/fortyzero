"""Asset template library with default parameters for different energy assets."""

from typing import Any

ASSET_TEMPLATES: dict[str, dict[str, Any]] = {
    "pv": {
        "depreciation_years": 20,
        "corporate_tax_rate": 0.30,  # Germany: ~30% effective (corp + trade + solidarity)
        "default_lifetime_years": 25,
        "typical_wacc": 0.04,
        "decommissioning_cost_per_kw": 50,  # €/kW
    },
    "wind": {
        "depreciation_years": 20,
        "corporate_tax_rate": 0.30,
        "default_lifetime_years": 25,
        "typical_wacc": 0.05,
        "decommissioning_cost_per_kw": 100,  # €/kW
    },
    "heat_network": {
        "depreciation_years": 40,
        "corporate_tax_rate": 0.30,
        "default_lifetime_years": 40,
        "typical_wacc": 0.06,
        "decommissioning_cost_per_kw": 20,  # €/kW
    },
    "chp": {
        "depreciation_years": 15,
        "corporate_tax_rate": 0.30,
        "default_lifetime_years": 20,
        "typical_wacc": 0.06,
        "decommissioning_cost_per_kw": 75,  # €/kW
    },
}
