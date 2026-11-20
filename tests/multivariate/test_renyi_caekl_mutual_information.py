"""
Tests for dit.multivariate.renyi_caekl_mutual_information.
"""

import numpy as np
import pytest

import dit
from dit.exceptions import ditException
from dit.multivariate import (
    caekl_mutual_information,
    renyi_caekl_mutual_information,
    renyi_total_correlation,
)

ORDERS = [0, 0.5, 1, 2, 5, np.inf]

DISTS = [
    dit.example_dists.Xor(),
    dit.example_dists.And(),
    dit.example_dists.RdnXor(),
    dit.example_dists.giant_bit(3, 2),
    dit.example_dists.giant_bit(4, 2),
    dit.Distribution([(0, 0, 0), (0, 1, 1), (1, 1, 0)], [0.5, 0.25, 0.25]),
]


@pytest.mark.parametrize("dist", DISTS)
def test_rcaekl_shannon_order1(dist):
    """
    Rényi CAEKL mutual information of order 1 matches the Shannon CAEKL mutual information.
    """
    assert renyi_caekl_mutual_information(dist, 1) == pytest.approx(caekl_mutual_information(dist))


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("order", ORDERS)
def test_rcaekl_bounded_by_total_correlation(dist, order):
    """
    The CAEKL mutual information is attained by some partition, so it is no larger
    than the singleton partition's value, T_a / (n - 1).
    """
    n = dist.outcome_length()
    singleton = renyi_total_correlation(dist, order) / (n - 1)
    assert renyi_caekl_mutual_information(dist, order) <= singleton + 1e-9


@pytest.mark.parametrize("order", ORDERS)
def test_rcaekl_independent(order):
    """
    Independent variables have a Rényi CAEKL mutual information of 0 for any order.
    """
    d = dit.Distribution(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)
    assert renyi_caekl_mutual_information(d, order) == pytest.approx(0.0)


@pytest.mark.parametrize("order", ORDERS)
def test_rcaekl_bivariate_is_total_correlation(order):
    """
    For two variables the only non-trivial partition is the singleton one, so CAEKL
    reduces to the total correlation.
    """
    d = dit.Distribution([(0, 0), (0, 1), (1, 1)], [0.5, 0.25, 0.25])
    rcaekl = renyi_caekl_mutual_information(d, order)
    assert rcaekl == pytest.approx(renyi_total_correlation(d, order))


@pytest.mark.parametrize("order", ORDERS)
def test_rcaekl_giant_bit(order):
    """
    Every variable of a giant bit is a copy of the others, so every non-trivial
    partition gives one shared bit regardless of order.
    """
    d = dit.example_dists.giant_bit(4, 2)
    assert renyi_caekl_mutual_information(d, order) == pytest.approx(1.0)


def test_rcaekl_rvs():
    """
    Test passing explicit rvs subsets.
    """
    d = dit.example_dists.Xor()
    assert renyi_caekl_mutual_information(d, 1, rvs=[[0], [1]]) == pytest.approx(0.0)
    assert renyi_caekl_mutual_information(d, 1, rvs=[[0], [1], [2]]) == pytest.approx(0.5)


def test_rcaekl_exceptions():
    """
    Test exception handling for invalid order, too few groups, and unsupported crvs.
    """
    d = dit.example_dists.Xor()

    with pytest.raises(ValueError):
        renyi_caekl_mutual_information(d, -1)

    with pytest.raises(ValueError):
        renyi_caekl_mutual_information(d, 1, rvs=[[0]])

    with pytest.raises(ditException):
        renyi_caekl_mutual_information(d, 1, crvs=[2])
