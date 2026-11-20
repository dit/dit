"""
Tests for dit.divergences.variational_distance
"""

import pytest

from dit import Distribution
from dit.divergences import (
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    variational_distance,
)
from dit.divergences.pmf import bhattacharyya_distance as bd_pmf

d1 = Distribution(["0", "1"], [1 / 2, 1 / 2])
d2 = Distribution(["0", "1"], [1 / 4, 3 / 4])
d3 = Distribution(["1", "2"], [1 / 4, 3 / 4])


def test_vd1():
    """
    Test against known value.
    """
    vd = variational_distance(d1, d2)
    assert vd == pytest.approx(0.25)


def test_vd2():
    """
    Test against known value.
    """
    vd = variational_distance(d1, d3)
    assert vd == pytest.approx(0.75)


def test_bc():
    """
    Test against known value.
    """
    bc = bhattacharyya_coefficient(d1, d2)
    assert bc == pytest.approx(0.9659258262890682)


def test_bd_identical():
    """
    Bhattacharyya distance is 0 for identical distributions.
    """
    bd = bhattacharyya_distance(d1, d1)
    assert bd == pytest.approx(0.0)


def test_bd_known():
    """
    Test against known value: -log2(BC(d1, d2)).
    """
    import numpy as np

    bc = bhattacharyya_coefficient(d1, d2)
    bd = bhattacharyya_distance(d1, d2)
    assert bd == pytest.approx(-np.log2(bc))


def test_bd_disjoint():
    """
    Bhattacharyya distance is infinity for completely disjoint distributions.
    """
    import numpy as np

    d_a = Distribution(["0"], [1.0])
    d_b = Distribution(["1"], [1.0])
    bd = bhattacharyya_distance(d_a, d_b)
    assert bd == np.inf


def test_bd_pmf():
    """
    Test pmf implementation directly.
    """
    import numpy as np

    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    bd = bd_pmf(p, q)
    assert bd == pytest.approx(-np.log2(np.sqrt(0.5 * 0.25) + np.sqrt(0.5 * 0.75)))
