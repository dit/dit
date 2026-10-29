"""
Tests for dit.divergences.maximal_cross_entropy.
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.divergences import cross_entropy, kullback_leibler_divergence, maximal_cross_entropy
from dit.divergences.maximal_cross_entropy import _max_coupling, _transport, maximal_cross_entropy_pmf
from dit.exceptions import ditException
from dit.shannon import entropy
from dit.utils.testing import distributions


def dsbs(p):
    """The doubly symmetric binary source with crossover `p`."""
    return Distribution(["00", "01", "10", "11"], [(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])


def product(a, b):
    """Independent bits with P(X=0) = a and P(Y=0) = b."""
    return Distribution(["00", "01", "10", "11"], [a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)])


@pytest.mark.parametrize("p", [0.1, 0.2, 0.4])
@pytest.mark.parametrize(("a", "b"), [(0.3, 0.6), (0.9, 0.2), (0.5, 0.5), (0.0, 0.7)])
def test_dsbs_closed_form(p, a, b):
    """
    Yu and Tan's Example 1: the coupling moves as much mass off the diagonal as
    the marginals allow.
    """
    a0, b0 = (1 - p) / 2, p / 2
    expected = np.log2(1 / a0) + min(a + b, 2 - a - b) * np.log2(a0 / b0)
    assert maximal_cross_entropy(product(a, b), dsbs(p)) == pytest.approx(expected)


def test_dsbs_self():
    """
    With uniform marginals every bit of mass goes off the diagonal.
    """
    assert maximal_cross_entropy(dsbs(0.2)) == pytest.approx(np.log2(10))


@settings(max_examples=25, deadline=None)
@given(dist=distributions(alphabets=(2, 3), zeros=False))
def test_entropy_bound(dist):
    """
    Yu and Tan's Proposition 2: it exceeds the joint entropy by at least the
    Jeffreys divergence between the distribution and the product of its marginals.
    """
    independent = dist.marginal([0]) @ dist.marginal([1])
    jeffreys = kullback_leibler_divergence(dist, independent) + kullback_leibler_divergence(independent, dist)
    assert maximal_cross_entropy(dist) >= entropy(dist) + jeffreys - 1e-9
    assert maximal_cross_entropy(dist) >= cross_entropy(independent, dist) - 1e-9


def test_independent():
    """
    The cost of an independent target separates, so every coupling has the same
    cross entropy, and the maximal cross entropy of the target with itself is its
    entropy.
    """
    q = product(0.4, 0.3)
    p = Distribution(["00", "01", "11"], [0.2, 0.5, 0.3])
    expected = cross_entropy(p.marginal([0]), q.marginal([0])) + cross_entropy(p.marginal([1]), q.marginal([1]))
    assert maximal_cross_entropy(p, q) == pytest.approx(expected)
    assert maximal_cross_entropy(q) == pytest.approx(entropy(q))


def test_infinite():
    """
    It is infinite when a coupling can place mass where the target vanishes, and
    finite when the supports keep clear of it.
    """
    erasure = Distribution(["00", "0e", "1e", "11"], [0.35, 0.15, 0.15, 0.35])
    assert maximal_cross_entropy(erasure) == np.inf
    p = Distribution(["00", "0e"], [0.5, 0.5])
    assert maximal_cross_entropy(p, erasure) == pytest.approx((np.log2(1 / 0.35) + np.log2(1 / 0.15)) / 2)


def test_groups():
    """
    Groups of variables couple as single variables, whatever their order.
    """
    d = Distribution(["000", "011", "101", "110", "111"], [0.1, 0.2, 0.3, 0.15, 0.25])
    value = maximal_cross_entropy(d, rvs=[[0, 2], [1]])
    assert maximal_cross_entropy(d, rvs=[[2, 0], [1]]) == pytest.approx(value)
    assert maximal_cross_entropy(d, rvs=[[1], [0, 2]]) == pytest.approx(value)


def test_base():
    """
    The value is reported in the base of the distribution.
    """
    d = dsbs(0.3)
    assert maximal_cross_entropy(d.copy(base=np.e)) == pytest.approx(maximal_cross_entropy(d) * np.log(2))


def test_rvs_errors():
    """
    Exactly two groups of variables are coupled.
    """
    d = Distribution(["000", "111"], [0.5, 0.5])
    with pytest.raises(ditException):
        maximal_cross_entropy(d)
    with pytest.raises(ditException):
        maximal_cross_entropy(d, rvs=[[0], [1], [2]])


def test_pmf():
    """
    The pmf form normalizes its marginals.
    """
    pxy = [[0.4, 0.1], [0.1, 0.4]]
    assert maximal_cross_entropy_pmf([1, 1], [2, 2], pxy) == pytest.approx(np.log2(10))
    assert maximal_cross_entropy_pmf([1, 1], [1, 0], [[0.5, 0.0], [0.0, 0.5]]) == np.inf


@pytest.mark.parametrize("shape", [(2, 4), (4, 2), (2, 2), (3, 3)])
def test_closed_form_matches_linear_program(shape):
    """
    The closed form used when one side is binary agrees with the linear program,
    and its dual potentials are feasible and tight.
    """
    rng = np.random.default_rng(0)
    m, n = shape
    cost = rng.uniform(0, 5, size=(m, n))
    a, b = rng.dirichlet(np.ones(m), 6), rng.dirichlet(np.ones(n), 6)
    values, f, g = _max_coupling(a, b, cost)
    plans, _, _ = _transport(a, b, np.broadcast_to(cost, (6, m, n)), maximize=True)
    assert values == pytest.approx((plans * cost).sum(axis=(1, 2)))
    assert (a * f).sum(axis=1) + (b * g).sum(axis=1) == pytest.approx(values)
    assert (f[:, :, np.newaxis] + g[:, np.newaxis, :] >= cost - 1e-9).all()
