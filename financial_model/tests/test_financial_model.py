"""Tests for the main FinancialModel class."""

import numpy as np
import pytest

from financial_model import FinancialModel


class TestFinancialModelInit:
    """Tests for FinancialModel initialization."""

    def test_init_valid_asset_type(self) -> None:
        """Test initialization with valid asset types."""
        for asset_type in ["pv", "wind", "heat_network", "chp"]:
            model = FinancialModel(asset_type=asset_type)
            assert model.asset_type == asset_type
            assert model.template is not None

    def test_init_invalid_asset_type(self) -> None:
        """Test initialization with invalid asset type raises error."""
        with pytest.raises(ValueError, match="Unknown asset_type"):
            FinancialModel(asset_type="invalid_type")

    def test_init_sub_models_created(self) -> None:
        """Test that all sub-models are instantiated."""
        model = FinancialModel(asset_type="pv")

        assert model.revenue_model is not None
        assert model.cost_model is not None
        assert model.tax_model is not None
        assert model.financing_model is not None
        assert model.dcf_engine is not None
        assert model.kpi_calculator is not None


class TestFinancialModelCalculate:
    """Tests for FinancialModel.calculate()."""

    @pytest.fixture
    def simple_params(self) -> dict:
        """Create simple test parameters."""
        return {
            "project": {
                "name": "Test Project",
                "asset_type": "pv",
                "lifetime_years": 5,
                "development_duration_years": 0,
                "construction_duration_years": 0,
            },
            "technical": {
                "capacity": 100,  # kW
            },
            "financial": {
                "capex": {
                    "development": [],
                    "construction": [
                        {
                            "phase": "construction",
                            "year": 0,
                            "month": 0,
                            "amount": 100000,
                            "category": "modules",
                        }
                    ],
                    "replacement": [],
                    "decommissioning": {"enabled": False},
                },
                "opex": {
                    "fixed": [
                        {
                            "category": "maintenance",
                            "annual_amount": 1000,
                            "indexed": False,
                        }
                    ],
                    "variable": [],
                },
                "revenue": {
                    "streams": [
                        {
                            "name": "fixed_tariff",
                            "type": "fixed_price",
                            "price_structure": {
                                "fixed_period": {
                                    "start_year": 0,
                                    "end_year": 4,
                                    "price": 0.10,  # €/kWh
                                    "indexed": False,
                                }
                            },
                        }
                    ]
                },
                "financing": {
                    "equity_share": 1.0,  # 100% equity
                },
                "tax": {
                    "corporate_tax_rate": 0.30,
                    "depreciation_years": 5,
                },
                "discount": {
                    "wacc": 0.05,
                },
                "inflation": {
                    "base_rate": 0.02,
                },
            },
        }

    @pytest.fixture
    def monthly_volumes(self) -> np.ndarray:
        """Create simple monthly volumes (5 years = 60 months)."""
        # 100 kW × 1000 kWh/kW/year ÷ 12 months = ~8333 kWh/month
        return np.full(60, 8333.0)

    def test_calculate_returns_expected_structure(
        self, simple_params: dict, monthly_volumes: np.ndarray
    ) -> None:
        """Test that calculate returns all expected keys."""
        model = FinancialModel(asset_type="pv")
        results = model.calculate(simple_params, monthly_volumes)

        # Check top-level keys
        assert "kpis" in results
        assert "cash_flows" in results
        assert "financing" in results

        # Check KPIs
        kpis = results["kpis"]
        assert "npv_project" in kpis
        assert "npv_equity" in kpis
        assert "irr_project" in kpis
        assert "irr_equity" in kpis
        assert "payback_simple" in kpis
        assert "payback_discounted" in kpis
        assert "dscr_min" in kpis
        assert "dscr_avg" in kpis
        assert "lcoe" in kpis

    def test_calculate_raises_without_volumes(self, simple_params: dict) -> None:
        """Test that calculate raises error without volumes or volume model."""
        model = FinancialModel(asset_type="pv")

        with pytest.raises(ValueError, match="monthly_volumes or volume_model"):
            model.calculate(simple_params)
