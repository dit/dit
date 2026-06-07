"""
Tests for dit.algorithms.ipf (Iterative Proportional Fitting).
"""

import numpy as np
import pytest

import dit
from dit.algorithms import ipf_dist, maxent_dist
from dit.algorithms.maxentropy import marginal_constraints_generic
from dit.algorithms.optutil import prepare_dist
from dit.multivariate import entropy as H


def residual(dist, structure):
    """Max abs deviation of the constrained marginals from the data marginals."""
    pd = prepare_dist(dist.copy())
    A, b = marginal_constraints_generic(pd, structure)
    return float(np.abs(A.dot(pd.pmf) - b).max())


@pytest.mark.parametrize(
    "structure",
    [
        [[0], [1], [2]],
        [[0, 1], [2]],
        [[0, 1], [1, 2]],
        [[0, 1], [0, 2], [1, 2]],
        [[0, 1, 2]],
    ],
)
def test_ipf_matches_scipy(structure):
    """IPF and the scipy optimizer should agree on the maxent distribution."""
    d = dit.example_dists.Xor()
    d_ipf = ipf_dist(d, structure)
    d_scipy = maxent_dist(d, structure, method="scipy")
    assert d_ipf.is_approx_equal(d_scipy, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "structure",
    [
        [[0], [1], [2]],
        [[0, 1], [2]],
        [[0, 1], [1, 2]],
        [[0, 1], [0, 2], [1, 2]],
    ],
)
def test_ipf_satisfies_constraints(structure):
    """The IPF reconstruction should match the data's constrained marginals."""
    rng = np.random.default_rng(0)
    outcomes = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]
    d = dit.Distribution(outcomes, rng.dirichlet(np.ones(8)))
    d_ipf = ipf_dist(d, structure)
    assert residual(d_ipf, structure) < 1e-9


def test_ipf_acyclic_closed_form():
    """For an acyclic structure, IPF reproduces p(A,B)p(B,C)/p(B)."""
    rng = np.random.default_rng(1)
    outcomes = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]
    d = dit.Distribution(outcomes, rng.dirichlet(np.ones(8)))
    d_ipf = ipf_dist(d, [[0, 1], [1, 2]])
    pAB = d.marginal([0, 1])
    pBC = d.marginal([1, 2])
    pB = d.marginal([1])
    for outcome, p in d_ipf.zipped():
        a, b, c = outcome
        expected = pAB[a + b] * pBC[b + c] / pB[b]
        assert p == pytest.approx(expected, abs=1e-9)


def test_ipf_default_for_maxent_dist():
    """maxent_dist should use IPF by default."""
    d = dit.example_dists.Xor()
    structure = [[0, 1], [1, 2]]
    assert maxent_dist(d, structure).is_approx_equal(ipf_dist(d, structure), rtol=1e-6, atol=1e-6)


def test_maxent_dist_bad_method():
    """An unknown method should raise."""
    d = dit.example_dists.Xor()
    with pytest.raises(ValueError):
        maxent_dist(d, [[0], [1], [2]], method="nope")


def test_ipf_handles_zeros():
    """A distribution with zero-probability outcomes should still converge."""
    d = dit.Distribution(["000", "111"], [0.5, 0.5])
    d_ipf = ipf_dist(d, [[0, 1], [1, 2]])
    assert H(d_ipf) == pytest.approx(1.0, abs=1e-6)
