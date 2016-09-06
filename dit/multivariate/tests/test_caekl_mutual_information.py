"""
Tests for dit.multivariate.caekl_mutual_information.
"""

from __future__ import division

import pytest

from dit import Distribution as D, ScalarDistribution as SD
from dit.distconst import all_dist_structures, random_distribution
from dit.multivariate import (caekl_mutual_information as J,
                              coinformation as I,
                              total_correlation as T,
                             )
from dit.exceptions import ditException

@pytest.mark.parametrize('d', list(all_dist_structures(2, 3)) +
                              [random_distribution(2, 3, alpha=(0.5,)*9) for _ in range(10)])
def test_caekl_1(d):
    """
    Ensure that it reduces to the mutual information for bivariate
    distributions.
    """
    assert I(d) == pytest.approx(J(d))

@pytest.mark.parametrize('d', list(all_dist_structures(3, 2)) +
                              [random_distribution(3, 2, alpha=(0.5,)*8) for _ in range(10)])
def test_caekl_2(d):
    """
    Ensure that it reduces to the mutual information for bivariate
    distributions reduced from multivariate.
    """
    rvs = [[0], [1]]
    assert I(d, rvs) == pytest.approx(J(d, rvs))

def test_caekl_3():
    """
    Test a known value.
    """
    d = D(['000', '011', '101', '110'], [1/4]*4)
    assert J(d) == pytest.approx(0.5)

@pytest.mark.parametrize('d', [random_distribution(4, 3, alpha=(0.5,)*3**4) for _ in range(10)])
def test_caekl_4(d):
    """
    Test that CAEKL is always less than or equal to the normalized total
    correlation.
    """
    assert J(d) <= (T(d)/3) + 1e-6
