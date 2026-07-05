"""
Tests for dit.divergences.earth_mover_distance.
"""

import warnings

import numpy as np
import pytest

from dit import Distribution
from dit.divergences.earth_movers_distance import (
    earth_movers_distance,
    earth_movers_distance_coupling,
    earth_movers_distance_pmf,
)


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
    sd1 = Distribution([0, 1, 2], [1 / 3] * 3)
    sd2 = Distribution([0, 1, 2], [1, 0, 0], trim=False)
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


def test_emd_coupling_numerical():
    """
    The coupling recovers the marginals and its expected cost is the EMD.
    """
    d1 = Distribution([0, 1, 2], [2 / 3, 1 / 6, 1 / 6])
    d2 = Distribution([0, 1, 2], [1 / 3, 1 / 3, 1 / 3])
    coupling = earth_movers_distance_coupling(d1, d2)

    marg1 = coupling.marginal([0])
    marg2 = coupling.marginal([1])
    assert marg1.pmf == pytest.approx(d1.pmf)
    assert marg2.pmf == pytest.approx(d2.pmf)

    cost = sum(abs(a - b) * p for (a, b), p in zip(coupling.outcomes, coupling.pmf, strict=True))
    assert cost == pytest.approx(earth_movers_distance(d1, d2))


def test_emd_coupling_categorical():
    """
    The coupling of categorical distributions recovers the marginals.
    """
    d1 = Distribution(["a", "b"], [2 / 3, 1 / 3])
    d2 = Distribution(["a", "b"], [0, 1], trim=False)
    coupling = earth_movers_distance_coupling(d1, d2)

    marg1 = coupling.marginal([0])
    marg2 = coupling.marginal([1])
    assert dict(zip(marg1.outcomes, marg1.pmf, strict=True)) == pytest.approx({("a",): 2 / 3, ("b",): 1 / 3})
    assert dict(zip(marg2.outcomes, marg2.pmf, strict=True)) == pytest.approx({("b",): 1.0})


def test_emd_coupling_distances():
    """
    An explicit distance matrix is respected by the coupling.
    """
    d1 = Distribution(["a", "b"], [2 / 3, 1 / 3])
    d2 = Distribution(["c", "d"], [0, 1], trim=False)
    distances = np.asarray([[0, 1], [1, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coupling = earth_movers_distance_coupling(d1, d2, distances=distances)
        emd = earth_movers_distance(d1, d2, distances=distances)

    marg1 = coupling.marginal([0])
    marg2 = coupling.marginal([1])
    assert dict(zip(marg1.outcomes, marg1.pmf, strict=True)) == pytest.approx({("a",): 2 / 3, ("b",): 1 / 3})
    assert dict(zip(marg2.outcomes, marg2.pmf, strict=True)) == pytest.approx({("d",): 1.0})

    cost = sum(
        distances[i][j] * coupling[(a, b)] for i, (a,) in enumerate(d1.outcomes) for j, (b,) in enumerate(d2.outcomes)
    )
    assert cost == pytest.approx(emd)
