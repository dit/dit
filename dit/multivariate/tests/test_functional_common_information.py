"""
Tests for dit.multivariate.functional_common_information.
"""

from __future__ import division

from nose.plugins.attrib import attr
from nose.tools import assert_almost_equal, assert_less_equal

from dit import Distribution, random_distribution
from dit.multivariate import (functional_common_information as F,
                              dual_total_correlation as B,
                              mss_common_information as M
                             )

def test_fci1():
    """
    Test known values.
    """
    d = Distribution(['000', '011', '101', '110'], [1/4]*4)
    assert_almost_equal(F(d), 2.0)
    assert_almost_equal(F(d, [[0], [1]]), 0.0)
    assert_almost_equal(F(d, [[0], [1]], [2]), 1.0)

def test_fci2():
    """
    Test known values w/ rv names.
    """
    d = Distribution(['000', '011', '101', '110'], [1/4]*4)
    d.set_rv_names('XYZ')
    assert_almost_equal(F(d), 2.0)
    assert_almost_equal(F(d, [[0], [1]]), 0.0)
    assert_almost_equal(F(d, [[0], [1]], [2]), 1.0)

@attr('slow')
def test_fci3():
    """
    Test that B <= F <= M.
    """
    dists = [ random_distribution(2, 3) for _ in range(3) ]
    for d in dists:
        b = B(d)
        f = F(d)
        m = M(d)
        yield assert_less_equal, b, f
        yield assert_less_equal, f, m

@attr('slow')
def test_fci4():
    """
    Test that B <= F <= M.
    """
    dists = [ random_distribution(3, 2) for _ in range(3) ]
    for d in dists:
        b = B(d)
        f = F(d)
        m = M(d)
        yield assert_less_equal, b, f
        yield assert_less_equal, f, m
