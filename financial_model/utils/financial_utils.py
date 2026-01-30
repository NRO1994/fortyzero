"""Financial utility functions for common calculations."""

import numpy as np


def calculate_annuity_payment(
    principal: float,
    interest_rate: float,
    term_years: int,
) -> float:
    """
    Calculate constant annuity payment for a loan.

    Uses the standard annuity formula:
    A = P × [r(1+r)^n] / [(1+r)^n - 1]

    Args:
        principal: Loan principal amount.
        interest_rate: Annual interest rate (e.g., 0.045 for 4.5%).
        term_years: Loan term in years.

    Returns:
        Annual annuity payment amount.

    Example:
        >>> calculate_annuity_payment(600000, 0.045, 15)
        55892.47  # approximately
    """
    if interest_rate == 0:
        return principal / term_years

    r = interest_rate
    n = term_years
    numerator = r * (1 + r) ** n
    denominator = (1 + r) ** n - 1

    return principal * (numerator / denominator)


def calculate_discount_factors(
    discount_rate: float,
    n_years: int,
    end_of_period: bool = True,
) -> np.ndarray:
    """
    Calculate discount factors for each year.

    Args:
        discount_rate: Discount rate (e.g., WACC).
        n_years: Number of years.
        end_of_period: If True, uses end-of-period convention (default).

    Returns:
        Array of discount factors for years 0 to n_years-1.

    Example:
        >>> calculate_discount_factors(0.05, 5)
        array([1.        , 0.95238095, 0.90702948, 0.86383760, 0.82270247])
    """
    years = np.arange(n_years)

    if end_of_period:
        # Year 0 not discounted, Year 1 discounted by 1/(1+r), etc.
        factors = 1 / (1 + discount_rate) ** years
        factors[0] = 1.0
    else:
        # Mid-period or beginning-of-period conventions could be added
        factors = 1 / (1 + discount_rate) ** years

    return factors
