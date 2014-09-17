"""
Tests for dit.shannon.shannon.
"""

from __future__ import division

from nose.tools import assert_almost_equal

from dit import Distribution as D, ScalarDistribution as SD
from dit.shannon import (entropy as H,
                         mutual_information as I,
                         conditional_entropy as CH,
                         entropy_pmf)

def test_H0():
    """ Test the entropy of a fair coin """
    d = [.5, .5]
    assert_almost_equal(entropy_pmf(d), 1.0)


def test_H1():
    """ Test the entropy of a fair coin """
    d = SD([1/2, 1/2])
    assert_almost_equal(H(d), 1.0)

def test_H2():
    """ Test the entropy of a fair coin, float style """
    assert_almost_equal(H(1/2), 1.0)

def test_H3():
    """ Test joint and marginal entropies """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert_almost_equal(H(d, [0]), 1.0)
    assert_almost_equal(H(d, [1]), 1.0)
    assert_almost_equal(H(d, [0, 1]), 2.0)
    assert_almost_equal(H(d), 2.0)

def test_H4():
    """ Test entropy in base 10 """
    d = SD([1/10]*10)
    d.set_base(10)
    assert_almost_equal(H(d), 1.0)

def test_I1():
    """ Test mutual information of independent variables """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [0], [1]), 0.0)

def test_I2():
    """ Test mutual information of dependent variables """
    outcomes = ['00', '11']
    pmf = [1/2]*2
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [0], [1]), 1.0)

def test_I3():
    """ Test mutual information of overlapping variables """
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [0, 1], [1, 2]), 2.0)

def test_CH1():
    """ Test conditional entropies """
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert_almost_equal(CH(d, [0], [1, 2]), 0.0)
    assert_almost_equal(CH(d, [0, 1], [2]), 1.0)
    assert_almost_equal(CH(d, [0], [0]), 0.0)
