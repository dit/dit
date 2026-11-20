"""
Tests for dit.multivariate.renyi_total_correlation.
"""

import numpy as np
import pytest

import dit
from dit.exceptions import ditException
from dit.multivariate import renyi_total_correlation, total_correlation


def test_rtc_shannon_order1():
    """
    Rényi total correlation of order 1 matches standard Shannon total correlation.
    """
    d = dit.example_dists.Xor()
    tc = total_correlation(d)
    rtc = renyi_total_correlation(d, 1)
    assert rtc == pytest.approx(tc)


def test_rtc_non_uniform():
    """
    Test Rényi total correlation on a non-uniform distribution across orders.
    """
    d = dit.Distribution([(0, 0), (0, 1), (1, 1)], [0.5, 0.25, 0.25])
    tc = total_correlation(d)
    rtc_1 = renyi_total_correlation(d, 1)
    assert rtc_1 == pytest.approx(tc)

    rtc_2 = renyi_total_correlation(d, 2)
    assert isinstance(rtc_2, float)
    assert rtc_2 > 0

    rtc_inf = renyi_total_correlation(d, np.inf)
    assert isinstance(rtc_inf, float)


def test_rtc_independent():
    """
    Independent variables have Rényi total correlation equal to 0 for any order.
    """
    d = dit.Distribution(["00", "01", "10", "11"], [0.25, 0.25, 0.25, 0.25])
    for order in [0, 0.5, 1, 2, 5, np.inf]:
        rtc = renyi_total_correlation(d, order)
        assert rtc == pytest.approx(0.0)


def test_rtc_rvs():
    """
    Test passing explicit rvs subsets.
    """
    d = dit.example_dists.Xor()
    # For Xor, any pair of variables is independent -> TC = 0
    rtc = renyi_total_correlation(d, 1, rvs=[[0], [1]])
    assert rtc == pytest.approx(0.0)

    # All three variables have TC = 1
    rtc_all = renyi_total_correlation(d, 1, rvs=[[0], [1], [2]])
    assert rtc_all == pytest.approx(1.0)


def test_rtc_exceptions():
    """
    Test exception handling for invalid order and unsupported crvs.
    """
    d = dit.example_dists.Xor()

    with pytest.raises(ValueError):
        renyi_total_correlation(d, -1)

    with pytest.raises(ditException):
        renyi_total_correlation(d, 1, crvs=[2])
