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

def test_fci3():
    """
    Test against known values
    """
    outcomes = ['000',
                'a00',
                '00c',
                'a0c',
                '011',
                'a11',
                '101',
                'b01',
                '01d',
                'a1d',
                '10d',
                'b0d',
                '110',
                'b10',
                '11c',
                'b1c',]
    pmf = [1/16]*16
    d = Distribution(outcomes, pmf)
    assert_almost_equal(F(d), 2.0)
