"""
Tests for dit.multivariate.cohesion.
"""

import pytest

from hypothesis import given

from dit.distconst import uniform
from dit.multivariate import cohesion, dual_total_correlation, total_correlation
from dit.utils.testing import distributions


reed_solomon = uniform([
    '0000', '0123', '0231', '0312',
    '1111', '1032', '1320', '1203',
    '2222', '2301', '2013', '2130',
    '3333', '3210', '3102', '3021',
])


@given(dist=distributions(alphabets=(2,)*4))
def test_cohesion_1(dist):
    """
    Test that k=1 is the total correlation.
    """
    tc = total_correlation(dist)
    c = cohesion(dist, k=1)
    assert tc == pytest.approx(c, abs=1e-4)


def test_cohesion_2():
    """
    Test that the reed-solomon distribution maximizes cohesion and connected
    information.
    """
    c = cohesion(reed_solomon, k=2)
    assert c == pytest.approx(12)


@given(dist=distributions(alphabets=(2,)*4))
def test_cohesion_3(dist):
    """
    Test that k=n-1 is the dual total correlation.
    """
    dtc = dual_total_correlation(dist)
    c = cohesion(dist, k=3)
    assert dtc == pytest.approx(c, abs=1e-4)
