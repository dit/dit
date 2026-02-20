"""
Tests for dit.divergences.earth_mover_distance.
"""

import warnings

import numpy as np
import pytest

from dit import Distribution, ScalarDistribution
from dit.divergences.earth_movers_distance import earth_movers_distance, earth_movers_distance_pmf


@pytest.mark.parametrize(
    ("p", "q", "emd"),
    [
        ([0, 1], [1, 0], 1),
        ([0.5, 0.5], [0, 1], 0.5),
        ([0.5, 0.5], [0.5, 0.5], 0),
        ([1, 0, 0], [0, 0, 1], 1),
    ],
)
def test_emd_pmf1(p, q, emd):
    """
    Test known examples.
    """
    emd2 = earth_movers_distance_pmf(p, q)
    assert emd2 == pytest.approx(emd, abs=1e-6)


def test_emd1():
    """ """
    sd1 = ScalarDistribution([0, 1, 2], [1 / 3] * 3)
    sd2 = ScalarDistribution([0, 1, 2], [1, 0, 0], trim=False)
    emd = earth_movers_distance(sd1, sd2)
    assert emd == pytest.approx(1.0)


def test_emd2():
    """ """
    d1 = Distribution(["a", "b"], [0, 1], trim=False)
    d2 = Distribution(["a", "b"], [1, 0], trim=False)
    emd = earth_movers_distance(d1, d2)
    assert emd == pytest.approx(1.0)


def test_emd3():
    """ """
    d1 = Distribution(["a", "b"], [2 / 3, 1 / 3])
    d2 = Distribution(["c", "d"], [0, 1], trim=False)
    distances = np.asarray([[0, 1], [1, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emd1 = earth_movers_distance(d1, d2)
        emd2 = earth_movers_distance(d1, d2, distances=distances)
    assert emd1 == pytest.approx(1.0)
    assert emd2 == pytest.approx(2 / 3)
