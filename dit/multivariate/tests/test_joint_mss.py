"""
Tests for dit.multivariate.joint_mss.
"""

from __future__ import division

from nose.tools import assert_almost_equal

from dit import Distribution
from dit.multivariate import joint_mss_entropy as M

def test_M1():
    """ Test M """
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
    assert_almost_equal(M(d), 2.0)

def test_M2():
    """ Test M with rv names """
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
    d.set_rv_names('XYZ')
    assert_almost_equal(M(d), 2.0)
