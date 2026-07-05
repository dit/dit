"""
Tests for dit.multivariate.s_information.
"""

import pytest

from dit.example_dists import giant_bit, n_mod_m
from dit.multivariate import (
    delta_k,
    dual_total_correlation,
    gamma_k,
    s_information,
    total_correlation,
)

d1 = giant_bit(5, 2)
d2 = n_mod_m(5, 2)


@pytest.mark.parametrize(
    ["dist", "rvs", "crvs", "value"],
    [
        (d1, [[0], [1], [2], [3], [4]], [], 5),
        (d1, [[0], [1], [2], [3]], [4], 0),
        (d1, [[0], [1], [2]], [3, 4], 0),
        (d2, [[0], [1], [2], [3], [4]], [], 5),
        (d2, [[0], [1], [2], [3]], [4], 4),
        (d2, [[0], [1], [2]], [3, 4], 3),
    ],
)
def test_s_information_1(dist, rvs, crvs, value):
    """
    Test the s-information against known values.
    """
    assert s_information(dist=dist, rvs=rvs, crvs=crvs) == pytest.approx(value)


@pytest.mark.parametrize("dist", [d1, d2])
def test_s_information_2(dist):
    """
    Test that the s-information equals the sum of the total correlation and the
    dual total correlation.
    """
    t = total_correlation(dist)
    b = dual_total_correlation(dist)
    assert s_information(dist) == pytest.approx(t + b)


@pytest.mark.parametrize("dist", [d1, d2])
def test_s_information_3(dist):
    """
    Test that the s-information equals both Delta^0 and Gamma^0.
    """
    s = s_information(dist)
    assert s == pytest.approx(delta_k(dist, 0))
    assert s == pytest.approx(gamma_k(dist, 0))
