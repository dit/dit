"""
Tests for dit.divergences.variational_distance
"""

import numpy as np
import pytest
from hypothesis import given

from dit import Distribution
from dit.divergences import (
    bhattacharyya_angle,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    fisher_rao_distance,
    hellinger_distance,
    hellinger_sum,
    matusita_distance,
    renyi_divergence,
    variational_distance,
)
from dit.divergences.pmf import bhattacharyya_distance as bd_pmf
from dit.utils.testing import distributions

d1 = Distribution(["0", "1"], [1 / 2, 1 / 2])
d2 = Distribution(["0", "1"], [1 / 4, 3 / 4])
d3 = Distribution(["1", "2"], [1 / 4, 3 / 4])

disjoint_a = Distribution(["0"], [1.0])
disjoint_b = Distribution(["1"], [1.0])

METRICS = [bhattacharyya_angle, fisher_rao_distance, matusita_distance]


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


def test_ba_known():
    """
    The Bhattacharyya angle is the arccosine of the Bhattacharyya coefficient.
    """
    assert bhattacharyya_angle(d1, d2) == pytest.approx(np.arccos(bhattacharyya_coefficient(d1, d2)))
    assert bhattacharyya_angle(d1, d2) == pytest.approx(np.pi / 12)


def test_frd_known():
    """
    The Fisher-Rao distance is twice the Bhattacharyya angle.
    """
    assert fisher_rao_distance(d1, d2) == pytest.approx(2 * bhattacharyya_angle(d1, d2))


def test_md_known():
    """
    The Matusita distance is sqrt(2 (1 - BC)), equivalently sqrt(2) times the Hellinger distance.
    """
    bc = bhattacharyya_coefficient(d1, d2)
    assert matusita_distance(d1, d2) == pytest.approx(np.sqrt(2 * (1 - bc)))
    assert matusita_distance(d1, d2) == pytest.approx(np.sqrt(2) * hellinger_distance(d1, d2))


@pytest.mark.parametrize("metric", METRICS)
def test_metrics_identical(metric):
    """
    Each of these is zero exactly when the distributions agree.
    """
    assert metric(d1, d1) == pytest.approx(0.0)


@pytest.mark.parametrize(("metric", "expected"), zip(METRICS, [np.pi / 2, np.pi, np.sqrt(2)], strict=True))
def test_metrics_disjoint(metric, expected):
    """
    Disjoint support puts each metric at its maximum.
    """
    assert metric(disjoint_a, disjoint_b) == pytest.approx(expected)


@pytest.mark.parametrize("metric", METRICS)
@given(
    dist1=distributions(alphabets=(4,)),
    dist2=distributions(alphabets=(4,)),
    dist3=distributions(alphabets=(4,)),
)
def test_metrics_triangle_inequality(metric, dist1, dist2, dist3):
    """
    Unlike the Bhattacharyya distance, these three are true metrics.
    """
    assert metric(dist1, dist3) <= metric(dist1, dist2) + metric(dist2, dist3) + 1e-9


@pytest.mark.parametrize("metric", METRICS)
@given(dist1=distributions(alphabets=(4,)), dist2=distributions(alphabets=(4,)))
def test_metrics_symmetry(metric, dist1, dist2):
    """
    Each of these is symmetric in its arguments.
    """
    assert metric(dist1, dist2) == pytest.approx(metric(dist2, dist1))


@given(dist1=distributions(alphabets=(4,)), dist2=distributions(alphabets=(4,)))
def test_bc_is_hellinger_sum_one_half(dist1, dist2):
    """
    The Bhattacharyya coefficient is the Hellinger sum at alpha = 1/2.
    """
    assert bhattacharyya_coefficient(dist1, dist2) == pytest.approx(hellinger_sum(dist1, dist2, alpha=0.5))


@given(dist1=distributions(alphabets=(4,), zeros=False), dist2=distributions(alphabets=(4,), zeros=False))
def test_bd_is_half_renyi_divergence_one_half(dist1, dist2):
    """
    The Bhattacharyya distance is half the Renyi divergence of order 1/2.
    """
    assert bhattacharyya_distance(dist1, dist2) == pytest.approx(renyi_divergence(dist1, dist2, alpha=0.5) / 2)
