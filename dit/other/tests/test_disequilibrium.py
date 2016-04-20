"""
Tests for dit.other.extropy.
"""

from __future__ import division

from itertools import combinations

from nose.tools import assert_almost_equal, assert_true, assert_greater

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
    assert_almost_equal(dis1, 0.43418979240387018)
    assert_almost_equal(dis2, 0.43418979240387018)
    assert_almost_equal(dis1, dis2)

def test_disequilibrium2():
    """
    Test that while the XOR distribution has non-zero disequilibrium, all its
    marginals have zero disequilibrium (are uniform).
    """
    assert_true(disequilibrium(d2) > 0)
    for rvs in combinations(flatten(d2.rvs), 2):
        assert_almost_equal(disequilibrium(d2, rvs), 0)

def test_disequilibrium3():
    """
    Test that uniform ScalarDistributions have zero disequilibrium.
    """
    for n in range(2, 11):
        d = uniform(n)
        yield assert_almost_equal, disequilibrium(d), 0

def test_disequilibrium4():
    """
    Test that uniform Distributions have zero disequilibrium.
    """
    for n in range(2, 11):
        d = Distribution.from_distribution(uniform(n))
        yield assert_almost_equal, disequilibrium(d), 0

def test_disequilibrium5():
    """
    Test that peaked ScalarDistributions have non-zero disequilibrium.
    """
    for n in range(2, 11):
        d = ScalarDistribution([1] + [0]*(n-1))
        yield assert_greater, disequilibrium(d), 0

def test_disequilibrium6():
    """
    Test that peaked Distributions have non-zero disequilibrium.
    """
    for n in range(2, 11):
        d = ScalarDistribution([1] + [0]*(n-1))
        d.make_dense()
        d = Distribution.from_distribution(d)
        yield assert_greater, disequilibrium(d), 0

def test_LMPR_complexity1():
    """
    Test LMPR complexity of known examples.
    """
    c1 = LMPR_complexity(d1)
    c2 = LMPR_complexity(d2)
    assert_almost_equal(c1, 0.28945986160258008)
    assert_almost_equal(c2, 0.28945986160258008)
    assert_almost_equal(c1, c2)

def test_LMPR_complexity2():
    """
    Test that uniform ScalarDistirbutions have zero complexity.
    """
    for n in range(2, 11):
        d = uniform(n)
        yield assert_almost_equal, LMPR_complexity(d), 0

def test_LMPR_complexity3():
    """
    Test that uniform Distirbutions have zero complexity.
    """
    for n in range(2, 11):
        d = Distribution.from_distribution(uniform(n))
        yield assert_almost_equal, LMPR_complexity(d), 0

def test_LMPR_complexity4():
    """
    Test that peaked ScalarDistributions have zero complexity.
    """
    for n in range(2, 11):
        d = ScalarDistribution([1] + [0]*(n-1))
        yield assert_almost_equal, LMPR_complexity(d), 0

def test_LMPR_complexity4():
    """
    Test that peaked Distributions have zero complexity.
    """
    for n in range(2, 11):
        d = ScalarDistribution([1] + [0]*(n-1))
        d.make_dense()
        d = Distribution.from_distribution(d)
        yield assert_almost_equal, LMPR_complexity(d), 0
