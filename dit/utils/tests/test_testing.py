"""
Tests for dit.util.testing
"""
from __future__ import division

import pytest

from hypothesis import find, given

import numpy as np

from dit.multivariate import coinformation, entropy
from dit.utils.testing import distributions, distribution_structures, markov_chains

@given(dist=distributions(alphabets=2))
def test_distributions1(dist):
    """
    A test for the distributions strategy.
    """
    assert entropy(dist) <= np.log2(len(dist.pmf))

@given(dist=distributions(alphabets=(2, 2)))
def test_distributions2(dist):
    """
    A test for the distributions strategy.
    """
    assert entropy(dist) <= np.log2(len(dist.pmf))

@given(dist=distributions(alphabets=((2, 2), (2, 2))))
def test_distributions3(dist):
    """
    A test for the distributions strategy.
    """
    assert entropy(dist) <= np.log2(len(dist.pmf))

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
