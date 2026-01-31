"""Tests for InflationCurve class."""

from pathlib import Path

import numpy as np
import pytest

from financial_model.models.inflation_curve import InflationCurve


class TestInflationCurveConstant:
    """Tests for constant inflation rate mode."""

    def test_constant_rate_initialization(self) -> None:
        """Test initialization with constant rate."""
        curve = InflationCurve({"base_rate": 0.02}, n_years=25)
        assert curve.get_rate(0) == pytest.approx(0.02)
        assert curve.get_rate(24) == pytest.approx(0.02)
        assert curve.is_constant

    def test_constant_rate_array(self) -> None:
        """Test get_rates returns uniform array."""
        curve = InflationCurve({"base_rate": 0.03}, n_years=10)
        rates = curve.get_rates()
        assert len(rates) == 10
        np.testing.assert_array_almost_equal(rates, np.full(10, 0.03))

    def test_boundary_years(self) -> None:
        """Test year boundary handling."""
        curve = InflationCurve({"base_rate": 0.02}, n_years=5)
        # Negative years should clamp to year 0
        assert curve.get_rate(-1) == pytest.approx(0.02)
        # Years beyond project length should clamp to last year
        assert curve.get_rate(10) == pytest.approx(0.02)

    def test_zero_inflation(self) -> None:
        """Test zero inflation rate."""
        curve = InflationCurve({"base_rate": 0.0}, n_years=5)
        assert curve.get_rate(0) == pytest.approx(0.0)
        assert curve.get_rate(4) == pytest.approx(0.0)
        assert curve.is_constant

    def test_negative_inflation(self) -> None:
        """Test negative inflation (deflation)."""
        curve = InflationCurve({"base_rate": -0.01}, n_years=5)
        assert curve.get_rate(0) == pytest.approx(-0.01)
        assert curve.get_rate(4) == pytest.approx(-0.01)

    def test_cumulative_factors_constant(self) -> None:
        """Test cumulative factors for constant rate."""
        curve = InflationCurve({"base_rate": 0.02}, n_years=5)
        factors = curve.get_cumulative_factors()

        assert len(factors) == 5
        # Year 0: 1.0
        assert factors[0] == pytest.approx(1.0)
        # Year 1: 1.02
        assert factors[1] == pytest.approx(1.02)
        # Year 2: 1.02^2 = 1.0404
        assert factors[2] == pytest.approx(1.0404)
        # Year 3: 1.02^3 = 1.061208
        assert factors[3] == pytest.approx(1.061208)
        # Year 4: 1.02^4 = 1.08243216
        assert factors[4] == pytest.approx(1.08243216)


class TestInflationCurveCSV:
    """Tests for CSV-based inflation rates."""

    @pytest.fixture
    def csv_file(self, tmp_path: Path) -> Path:
        """Create temporary CSV file with inflation rates."""
        csv_path = tmp_path / "inflation.csv"
        csv_path.write_text(
            "year,rate\n"
            "0,0.020\n"
            "1,0.025\n"
            "2,0.030\n"
            "3,0.028\n"
            "4,0.026\n"
        )
        return csv_path

    def test_csv_loading(self, csv_file: Path) -> None:
        """Test loading rates from CSV."""
        curve = InflationCurve({"csv_path": str(csv_file)}, n_years=5)
        assert curve.get_rate(0) == pytest.approx(0.020)
        assert curve.get_rate(1) == pytest.approx(0.025)
        assert curve.get_rate(2) == pytest.approx(0.030)
        assert curve.get_rate(3) == pytest.approx(0.028)
        assert curve.get_rate(4) == pytest.approx(0.026)

    def test_csv_extrapolation(self, csv_file: Path) -> None:
        """Test extrapolation beyond CSV data."""
        curve = InflationCurve({"csv_path": str(csv_file)}, n_years=10)
        # Years 5-9 should use last value (0.026)
        assert curve.get_rate(5) == pytest.approx(0.026)
        assert curve.get_rate(9) == pytest.approx(0.026)

    def test_csv_truncation(self, csv_file: Path) -> None:
        """Test truncation when CSV has more data than needed."""
        curve = InflationCurve({"csv_path": str(csv_file)}, n_years=3)
        rates = curve.get_rates()
        assert len(rates) == 3
        np.testing.assert_array_almost_equal(rates, [0.020, 0.025, 0.030])

    def test_csv_not_found(self) -> None:
        """Test error when CSV file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            InflationCurve({"csv_path": "/nonexistent/file.csv"}, n_years=5)

    def test_csv_alternative_column_name(self, tmp_path: Path) -> None:
        """Test loading with alternative column name."""
        csv_path = tmp_path / "inflation2.csv"
        csv_path.write_text(
            "year,inflation_rate\n"
            "0,0.015\n"
            "1,0.018\n"
        )
        curve = InflationCurve(
            {"csv_path": str(csv_path), "rate_column": "inflation_rate"},
            n_years=2
        )
        assert curve.get_rate(0) == pytest.approx(0.015)
        assert curve.get_rate(1) == pytest.approx(0.018)

    def test_csv_auto_detect_column(self, tmp_path: Path) -> None:
        """Test auto-detection of common column names."""
        csv_path = tmp_path / "inflation3.csv"
        csv_path.write_text(
            "year,inflation\n"
            "0,0.022\n"
            "1,0.024\n"
        )
        # Should auto-detect 'inflation' column
        curve = InflationCurve({"csv_path": str(csv_path)}, n_years=2)
        assert curve.get_rate(0) == pytest.approx(0.022)

    def test_is_constant_false_for_varying(self, csv_file: Path) -> None:
        """Test is_constant returns False for varying rates."""
        curve = InflationCurve({"csv_path": str(csv_file)}, n_years=5)
        assert not curve.is_constant

    def test_cumulative_factors_varying(self, csv_file: Path) -> None:
        """Test cumulative factors for time-varying rates."""
        curve = InflationCurve({"csv_path": str(csv_file)}, n_years=5)
        factors = curve.get_cumulative_factors()

        assert len(factors) == 5
        # Year 0: 1.0
        assert factors[0] == pytest.approx(1.0)
        # Year 1: 1.020
        assert factors[1] == pytest.approx(1.020)
        # Year 2: 1.020 * 1.025 = 1.0455
        assert factors[2] == pytest.approx(1.0455)
        # Year 3: 1.020 * 1.025 * 1.030 = 1.07687
        assert factors[3] == pytest.approx(1.076865)
        # Year 4: 1.020 * 1.025 * 1.030 * 1.028 = 1.107
        assert factors[4] == pytest.approx(1.107017, rel=1e-4)


class TestInflationCurveValidation:
    """Tests for input validation."""

    def test_missing_config_key(self) -> None:
        """Test error when neither base_rate nor csv_path provided."""
        with pytest.raises(ValueError, match="must contain"):
            InflationCurve({}, n_years=5)

    def test_invalid_column_name(self, tmp_path: Path) -> None:
        """Test error when specified column doesn't exist."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("year,other_column\n0,0.02\n")

        with pytest.raises(ValueError, match="not found in CSV"):
            InflationCurve(
                {"csv_path": str(csv_path), "rate_column": "nonexistent"},
                n_years=5
            )

    def test_empty_inflation_config(self) -> None:
        """Test error with empty config."""
        with pytest.raises(ValueError):
            InflationCurve({}, n_years=5)


class TestInflationCurveIntegration:
    """Integration tests for InflationCurve with financial calculations."""

    def test_constant_rate_escalation(self) -> None:
        """Test that constant rate produces expected escalation."""
        curve = InflationCurve({"base_rate": 0.02}, n_years=5)
        factors = curve.get_cumulative_factors()

        # Apply to base amount
        base_amount = 1000.0
        escalated = base_amount * factors

        # Expected: 1000, 1020, 1040.4, 1061.208, 1082.43216
        expected = np.array([
            1000 * (1.02 ** i) for i in range(5)
        ])
        np.testing.assert_array_almost_equal(escalated, expected)

    def test_varying_rate_escalation(self, tmp_path: Path) -> None:
        """Test that varying rates produce expected escalation."""
        csv_path = tmp_path / "inflation.csv"
        csv_path.write_text(
            "year,rate\n"
            "0,0.02\n"
            "1,0.03\n"
            "2,0.04\n"
            "3,0.02\n"
            "4,0.02\n"
        )
        curve = InflationCurve({"csv_path": str(csv_path)}, n_years=5)
        factors = curve.get_cumulative_factors()

        # Apply to base amount
        base_amount = 1000.0
        escalated = base_amount * factors

        # Expected:
        # Year 0: 1000 (base)
        # Year 1: 1000 * 1.02 = 1020
        # Year 2: 1000 * 1.02 * 1.03 = 1050.6
        # Year 3: 1000 * 1.02 * 1.03 * 1.04 = 1092.624
        # Year 4: 1000 * 1.02 * 1.03 * 1.04 * 1.02 = 1114.47648
        expected = np.array([1000, 1020, 1050.6, 1092.624, 1114.47648])
        np.testing.assert_array_almost_equal(escalated, expected, decimal=2)
