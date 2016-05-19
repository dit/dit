"""
Tests for dit.multivariate.caekl_mutual_information.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_less_equal

from dit import Distribution as D, ScalarDistribution as SD
from dit.distconst import all_dist_structures, random_distribution
from dit.multivariate import (caekl_mutual_information as J,
                              coinformation as I,
                              total_correlation as T,
                             )
from dit.exceptions import ditException

def test_caekl_1():
    """
    Ensure that it reduces to the mutual information for bivariate
    distributions.
    """
    for d in all_dist_structures(2, 3):
        yield assert_almost_equal, I(d), J(d)
    for d in [random_distribution(2, 3, alpha=(0.5,)*9) for _ in range(10)]:
        yield assert_almost_equal, I(d), J(d)

def test_caekl_2():
    """
    Ensure that it reduces to the mutual information for bivariate
    distributions reduced from multivariate.
    """
    rvs = [[0], [1]]
    for d in all_dist_structures(3, 2):
        yield assert_almost_equal, I(d, rvs), J(d, rvs)
    for d in [random_distribution(3, 2, alpha=(0.5,)*8) for _ in range(10)]:
        yield assert_almost_equal, I(d, rvs), J(d, rvs)

def test_caekl_3():
    """
    Test a known value.
    """
    d = D(['000', '011', '101', '110'], [1/4]*4)
    assert_almost_equal(J(d), 0.5)

def test_caekl_4():
    """
    Test that CAEKL is always less than or equal to the normalized total
    correlation.
    """
    for d in [random_distribution(4, 3, alpha=(0.5,)*3**4) for _ in range(10)]:
        yield assert_less_equal, J(d), T(d)/3 + 1e-7
