"""
Tests for dit.util.testing
"""
from __future__ import division

import pytest

from hypothesis import find

import numpy as np

from dit.multivariate import coinformation
from dit.utils.testing import distributions, distribution_structures, markov_chains


def test_distributions1():
    """
    A test for the distributions strategy.
    """
    dist = distributions(alphabets=1).example()
    assert dist.alphabet == ((0, 1),)


def test_distributions2():
    """
    A test for the distributions strategy.
    """
    dist = distributions(alphabets=(2, 2)).example()
    assert dist.alphabet == ((0, 1), (0, 1))


def test_distributions3():
    """
    A test for the distributions strategy.
    """
    dist = distributions(alphabets=((2, 2), (2, 2))).example()
    assert dist.alphabet == ((0, 1), (0, 1))


def test_distribution_structures1():
    """
    A test for the distributions strategy.
    """
    dist = distribution_structures(size=1, alphabet=2).example()
    assert dist.alphabet == ((0, 1),)


def test_distribution_structures2():
    """
    A test for the distributions strategy.
    """
    dist = distribution_structures(size=(2, 4), alphabet=2).example()
    assert dist.outcome_length() in [2, 3, 4]


def test_distribution_structures3():
    """
    A test for the distributions strategy.
    """
    dist = distribution_structures(size=2, alphabet=(2, 4), uniform=False).example()
    assert dist.outcome_length() == 2
    assert set(dist.alphabet[0]) <= {0, 1, 2, 3}


def test_distribution_structures4():
    """
    A test for the distributions strategy.
    """
    dist = distribution_structures(size=(2, 3), alphabet=(2, 4), uniform=False).example()
    assert dist.outcome_length() in [2, 3]
    assert set(dist.alphabet[0]) <= {0, 1, 2, 3}


def predicate(d):
    return coinformation(d) <= -1/2


@pytest.mark.flaky(reruns=5)
def test_distribution_structures1():
    """
    A test for the distribution_structures strategy.
    """
    dists = distribution_structures(size=3, alphabet=2, uniform=False)
    example = find(dists, predicate)
    assert coinformation(example) <= -1/2


@pytest.mark.flaky(reruns=5)
def test_distribution_structures2():
    """
    A test for the distribution_structures strategy.
    """
    dists = distribution_structures(size=3, alphabet=2, uniform=True)
    example = find(dists, predicate)
    assert coinformation(example) <= -1/2


def test_markov_chains1():
    """
    Test the markov_chains strategy.
    """
    dist = markov_chains(alphabets=3).example()
    assert coinformation(dist, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-7)


def test_markov_chains2():
    """
    Test the markov_chains strategy.
    """
    dist = markov_chains(alphabets=(2, 2, 2)).example()
    assert coinformation(dist, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-7)


def test_markov_chains3():
    """
    Test the markov_chains strategy.
    """
    dist = markov_chains(alphabets=((2, 4), (2, 4), (2, 4))).example()
    assert coinformation(dist, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-7)
