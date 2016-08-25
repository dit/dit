"""
Tests for dit.other.extropy.
"""

from __future__ import division

from itertools import combinations

import pytest

from dit import Distribution, ScalarDistribution
from dit.example_dists import uniform
from dit.other import disequilibrium, LMPR_complexity
from dit.utils import flatten

d1 = Distribution(['000', '001', '110', '111'], [1/4]*4)
d2 = Distribution(['000', '011', '101', '110'], [1/4]*4)

def test_disequilibrium1():
    """
    Test that two known distributions have the same disequilibrium.
    """
    dis1 = disequilibrium(d1)
    dis2 = disequilibrium(d2)
    assert dis1 == pytest.approx(0.43418979240387018)
    assert dis2 == pytest.approx(0.43418979240387018)
    assert dis1 == pytest.approx(dis2)

def test_disequilibrium2():
    """
    Test that while the XOR distribution has non-zero disequilibrium, all its
    marginals have zero disequilibrium (are uniform).
    """
    assert disequilibrium(d2) > 0
    for rvs in combinations(flatten(d2.rvs), 2):
        assert disequilibrium(d2, rvs) == pytest.approx(0)

@pytest.mark.parametrize('n', range(2, 11))
def test_disequilibrium3(n):
    """
    Test that uniform ScalarDistributions have zero disequilibrium.
    """
    d = uniform(n)
    assert disequilibrium(d) == pytest.approx(0)

@pytest.mark.parametrize('n', range(2, 11))
def test_disequilibrium4(n):
    """
    Test that uniform Distributions have zero disequilibrium.
    """
    d = Distribution.from_distribution(uniform(n))
    assert disequilibrium(d) == pytest.approx(0)

@pytest.mark.parametrize('n', range(2, 11))
def test_disequilibrium5(n):
    """
    Test that peaked ScalarDistributions have non-zero disequilibrium.
    """
    d = ScalarDistribution([1] + [0]*(n-1))
    assert disequilibrium(d) >= 0

@pytest.mark.parametrize('n', range(2, 11))
def test_disequilibrium6(n):
    """
    Test that peaked Distributions have non-zero disequilibrium.
    """
    d = ScalarDistribution([1] + [0]*(n-1))
    d.make_dense()
    d = Distribution.from_distribution(d)
    assert disequilibrium(d) >= 0

def test_LMPR_complexity1():
    """
    Test LMPR complexity of known examples.
    """
    c1 = LMPR_complexity(d1)
    c2 = LMPR_complexity(d2)
    assert c1 == pytest.approx(0.28945986160258008)
    assert c2 == pytest.approx(0.28945986160258008)
    assert c1 == pytest.approx(c2)

@pytest.mark.parametrize('n', range(2, 11))
def test_LMPR_complexity2(n):
    """
    Test that uniform ScalarDistirbutions have zero complexity.
    """
    d = uniform(n)
    assert LMPR_complexity(d) == pytest.approx(0)

@pytest.mark.parametrize('n', range(2, 11))
def test_LMPR_complexity3(n):
    """
    Test that uniform Distirbutions have zero complexity.
    """
    d = Distribution.from_distribution(uniform(n))
    assert LMPR_complexity(d) == pytest.approx(0)

@pytest.mark.parametrize('n', range(2, 11))
def test_LMPR_complexity4(n):
    """
    Test that peaked ScalarDistributions have zero complexity.
    """
    d = ScalarDistribution([1] + [0]*(n-1))
    assert LMPR_complexity(d) == pytest.approx(0)

@pytest.mark.parametrize('n', range(2, 11))
def test_LMPR_complexity4(n):
    """
    Test that peaked Distributions have zero complexity.
    """
    d = ScalarDistribution([1] + [0]*(n-1))
    d.make_dense()
    d = Distribution.from_distribution(d)
    assert LMPR_complexity(d) == pytest.approx(0)
