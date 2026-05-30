"""
Tests for dit.multivariate.delta_gamma.
"""

import pytest

from dit.example_dists import giant_bit, n_mod_m
from dit.multivariate import (
    delta_k,
    dual_total_correlation,
    gamma_k,
    o_information,
    total_correlation,
)

d1 = giant_bit(5, 2)
d2 = n_mod_m(5, 2)


@pytest.mark.parametrize(
    ["dist", "k", "value"],
    [
        (d1, 0, 5),
        (d1, 1, 1),
        (d1, 2, -3),
        (d2, 0, 5),
        (d2, 1, 4),
        (d2, 2, 3),
    ],
)
def test_delta_k(dist, k, value):
    """
    Test the Delta^k measure against known values.
    """
    assert delta_k(dist, k) == pytest.approx(value)


@pytest.mark.parametrize(
    ["dist", "k", "value"],
    [
        (d1, 0, 5),
        (d1, 1, 4),
        (d1, 2, 3),
        (d2, 0, 5),
        (d2, 1, 1),
        (d2, 2, -3),
    ],
)
def test_gamma_k(dist, k, value):
    """
    Test the Gamma^k measure against known values.
    """
    assert gamma_k(dist, k) == pytest.approx(value)


@pytest.mark.parametrize("dist", [d1, d2])
def test_delta_k_special_cases(dist):
    """
    Delta^1 is the dual total correlation, and Delta^2 is the negative
    O-information.
    """
    assert delta_k(dist, 1) == pytest.approx(dual_total_correlation(dist))
    assert delta_k(dist, 2) == pytest.approx(-o_information(dist))


@pytest.mark.parametrize("dist", [d1, d2])
def test_gamma_k_special_cases(dist):
    """
    Gamma^1 is the total correlation, and Gamma^2 is the O-information.
    """
    assert gamma_k(dist, 1) == pytest.approx(total_correlation(dist))
    assert gamma_k(dist, 2) == pytest.approx(o_information(dist))


def test_conditional():
    """
    Test that conditioning is threaded through to the underlying measures.
    """
    rvs = [[0], [1], [2], [3]]
    crvs = [4]
    t = total_correlation(d2, rvs=rvs, crvs=crvs)
    b = dual_total_correlation(d2, rvs=rvs, crvs=crvs)
    assert delta_k(d2, 2, rvs=rvs, crvs=crvs) == pytest.approx(b - t)
    assert gamma_k(d2, 2, rvs=rvs, crvs=crvs) == pytest.approx(t - b)
