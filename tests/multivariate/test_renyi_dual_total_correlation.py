"""
Tests for dit.multivariate.renyi_dual_total_correlation.
"""

import numpy as np
import pytest

import dit
from dit.exceptions import ditException
from dit.multivariate import (
    dual_total_correlation,
    renyi_dual_total_correlation,
    renyi_total_correlation,
)
from dit.other import renyi_entropy

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
def test_rdtc_shannon_order1(dist):
    """
    Rényi dual total correlation of order 1 matches the Shannon dual total correlation.
    """
    assert renyi_dual_total_correlation(dist, 1) == pytest.approx(dual_total_correlation(dist))


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("order", ORDERS)
def test_rdtc_matches_definition(dist, order):
    """
    The total-correlation identity agrees with the direct sum of joint Rényi entropies,
    sum_i H_a(X_{-i}) - (n - 1) H_a(X).
    """
    rvs = [[i] for i in range(dist.outcome_length())]
    n = len(rvs)
    others = (set().union(*(rvs[:i] + rvs[i + 1 :])) for i in range(n))
    expected = sum(renyi_entropy(dist, order, rv) for rv in others)
    expected -= (n - 1) * renyi_entropy(dist, order, set().union(*rvs))

    assert renyi_dual_total_correlation(dist, order) == pytest.approx(expected)


@pytest.mark.parametrize("order", ORDERS)
def test_rdtc_independent(order):
    """
    Independent variables have a Rényi dual total correlation of 0 for any order.
    """
    d = dit.Distribution(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)
    assert renyi_dual_total_correlation(d, order) == pytest.approx(0.0)


@pytest.mark.parametrize("order", ORDERS)
def test_rdtc_bivariate_is_total_correlation(order):
    """
    For two variables the dual total correlation equals the total correlation.
    """
    d = dit.Distribution([(0, 0), (0, 1), (1, 1)], [0.5, 0.25, 0.25])
    rdtc = renyi_dual_total_correlation(d, order)
    assert rdtc == pytest.approx(renyi_total_correlation(d, order))


@pytest.mark.parametrize("order", ORDERS)
def test_rdtc_single_variable(order):
    """
    A single variable has no dual total correlation.
    """
    d = dit.example_dists.Xor()
    assert renyi_dual_total_correlation(d, order, rvs=[[0]]) == pytest.approx(0.0)


def test_rdtc_rvs():
    """
    Test passing explicit rvs subsets.
    """
    d = dit.example_dists.Xor()
    # Any pair of Xor's variables is independent.
    assert renyi_dual_total_correlation(d, 1, rvs=[[0], [1]]) == pytest.approx(0.0)
    # All three are pairwise independent but jointly determined.
    assert renyi_dual_total_correlation(d, 1, rvs=[[0], [1], [2]]) == pytest.approx(2.0)


def test_rdtc_exceptions():
    """
    Test exception handling for invalid order and unsupported crvs.
    """
    d = dit.example_dists.Xor()

    with pytest.raises(ValueError):
        renyi_dual_total_correlation(d, -1)

    with pytest.raises(ditException):
        renyi_dual_total_correlation(d, 1, crvs=[2])
