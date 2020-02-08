"""
Tests for dit.util.testing
"""

import pytest

from hypothesis import find, given

from dit.multivariate import coinformation
from dit.utils.testing import distributions, distribution_structures, markov_chains


@given(dist=distributions(alphabets=1))
def test_distributions1(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.alphabet == ((0, 1),)


@given(dist=distributions(alphabets=(2, 2)))
def test_distributions2(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.alphabet == ((0, 1), (0, 1))


@given(dist=distributions(alphabets=((2, 2), (2, 2))))
def test_distributions3(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.alphabet == ((0, 1), (0, 1))


@given(dist=distribution_structures(size=1, alphabet=2))
def test_distribution_structures1(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.alphabet == ((0, 1),)


@given(dist=distribution_structures(size=(2, 4), alphabet=2))
def test_distribution_structures2(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.outcome_length() in [2, 3, 4]


@given(dist=distribution_structures(size=2, alphabet=(2, 4), uniform=False))
def test_distribution_structures3(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.outcome_length() == 2
    assert set(dist.alphabet[0]) <= {0, 1, 2, 3}


@given(dist=distribution_structures(size=(2, 3), alphabet=(2, 4), uniform=False))
def test_distribution_structures4(dist):
    """
    A test for the distributions strategy.
    """
    assert dist.outcome_length() in [2, 3]
    assert set(dist.alphabet[0]) <= {0, 1, 2, 3}


def predicate(d):
    return coinformation(d) <= -1 / 2


@pytest.mark.flaky(reruns=5)
def test_distribution_structures5():
    """
    A test for the distribution_structures strategy.
    """
    dists = distribution_structures(size=3, alphabet=2, uniform=False)
    example = find(dists, predicate)
    assert coinformation(example) <= -1 / 2


@pytest.mark.flaky(reruns=5)
def test_distribution_structures6():
    """
    A test for the distribution_structures strategy.
    """
    dists = distribution_structures(size=3, alphabet=2, uniform=True)
    example = find(dists, predicate)
    assert coinformation(example) <= -1 / 2


@given(dist=markov_chains(alphabets=3))
def test_markov_chains1(dist):
    """
    Test the markov_chains strategy.
    """
    assert coinformation(dist, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-7)


@given(dist=markov_chains(alphabets=(2, 2, 2)))
def test_markov_chains2(dist):
    """
    Test the markov_chains strategy.
    """
    assert coinformation(dist, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-7)


@given(dist=markov_chains(alphabets=((2, 4), (2, 4), (2, 4))))
def test_markov_chains3(dist):
    """
    Test the markov_chains strategy.
    """
    assert coinformation(dist, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-7)
