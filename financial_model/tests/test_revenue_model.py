"""Tests for the RevenueModel."""

import numpy as np
import pytest

from financial_model.models.revenue_model import RevenueModel


class TestRevenueModel:
    """Tests for RevenueModel calculations."""

    @pytest.fixture
    def revenue_model(self) -> RevenueModel:
        """Create RevenueModel instance."""
        return RevenueModel()

    @pytest.fixture
    def simple_volumes(self) -> np.ndarray:
        """Create simple monthly volumes (2 years = 24 months)."""
        return np.full(24, 1000.0)  # 1000 kWh/month

    def test_fixed_price_revenue(
        self, revenue_model: RevenueModel, simple_volumes: np.ndarray
    ) -> None:
        """Test revenue calculation with fixed price."""
        config = {
            "streams": [
                {
                    "name": "fixed_tariff",
                    "type": "fixed_price",
                    "price_structure": {
                        "fixed_period": {
                            "start_year": 0,
                            "end_year": 1,
                            "price": 0.10,  # €/kWh
                            "indexed": False,
                        }
                    },
                }
            ]
        }

        revenue = revenue_model.calculate(
            volumes=simple_volumes,
            revenue_config=config,
            inflation_rate=0.02,
            price_curve=None,
            n_months=24,
        )

        # Expected: 1000 kWh × €0.10/kWh = €100/month
        expected = np.full(24, 100.0)
        np.testing.assert_array_almost_equal(revenue, expected)

    def test_indexed_price_escalation(
        self, revenue_model: RevenueModel, simple_volumes: np.ndarray
    ) -> None:
        """Test revenue with price escalation."""
        config = {
            "streams": [
                {
                    "name": "indexed_tariff",
                    "type": "fixed_price",
                    "price_structure": {
                        "fixed_period": {
                            "start_year": 0,
                            "end_year": 1,
                            "price": 0.10,
                            "indexed": True,
                            "escalation_rate": 0.05,  # 5% escalation
                        }
                    },
                }
            ]
        }

        revenue = revenue_model.calculate(
            volumes=simple_volumes,
            revenue_config=config,
            inflation_rate=0.02,
            price_curve=None,
            n_months=24,
        )

        # Year 0: €0.10/kWh × 1000 kWh = €100
        # Year 1: €0.10 × 1.05 × 1000 = €105
        assert revenue[0] == pytest.approx(100.0)
        assert revenue[12] == pytest.approx(105.0)

    def test_empty_streams(
        self, revenue_model: RevenueModel, simple_volumes: np.ndarray
    ) -> None:
        """Test with no revenue streams returns zeros."""
        config = {"streams": []}

        revenue = revenue_model.calculate(
            volumes=simple_volumes,
            revenue_config=config,
            inflation_rate=0.02,
            price_curve=None,
            n_months=24,
        )

        np.testing.assert_array_equal(revenue, np.zeros(24))
