"""
Tests for dit.util.testing
"""
from __future__ import division

import pytest

from hypothesis import find

from dit.multivariate import coinformation
from dit.utils.testing import distributions

def predicate(d):
    return coinformation(d) <= -1/2

@pytest.mark.flaky(reruns=5)
def test_distributions1():
    """
    A test for the distributions strategy.
    """
    dists = distributions(alphabets=3)
    example = find(dists, predicate)
    assert coinformation(example) <= -1/2

@pytest.mark.flaky(reruns=5)
def test_distributions2():
    """
    A test for the distributions strategy.
    """
    dists = distributions(alphabets=(2, 2, 2))
    example = find(dists, predicate)
    assert coinformation(example) <= -1/2

@pytest.mark.flaky(reruns=5)
def test_distributions3():
    """
    A test for the distributions strategy.
    """
    dists = distributions(alphabets=((2, 3), (2, 3), (2, 3)))
    example = find(dists, predicate)
    assert coinformation(example) <= -1/2
