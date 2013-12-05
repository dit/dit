"""
Tests for dit.multivariate.binding_information.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

from dit import Distribution as D, ScalarDistribution as SD
from dit.multivariate import (binding_information as B,
                            residual_entropy as R)
from dit.shannon import (entropy as H,
                         mutual_information as I)
from dit.exceptions import ditException

def test_B1():
    """ Test B for two dependent variables """
    d = D(['00', '11'], [1/2, 1/2])
    assert_almost_equal(B(d), 1)

def test_B2():
    """ Test B for three dependent variables """
    d = D(['000', '111'], [1/2, 1/2])
    assert_almost_equal(B(d), 1)
    assert_almost_equal(B(d, [[0], [1]], [2]), 0)

def test_B3():
    """ Test B for four dependent variables """
    d = D(['0000', '1111'], [1/2, 1/2])
    assert_almost_equal(B(d), 1)
    assert_almost_equal(B(d, [[0], [1, 2]], [3]), 0)

def test_B4():
    """ Test B for xor distribution """
    d = D(['000', '011', '101', '110'], [1/4]*4)
    assert_almost_equal(B(d), 2)
    assert_almost_equal(B(d, [[0], [1], [2]]), 2)
    assert_almost_equal(B(d, [[0], [1]], [2]), 1)
    assert_almost_equal(B(d, [[0], [2]], [1]), 1)
    assert_almost_equal(B(d, [[1], [2]], [0]), 1)

def test_B5():
    """ Test B = I for two variables """
    d = D(['00', '01', '11'], [1/3]*3)
    assert_almost_equal(B(d), I(d, [0], [1]))

def test_B6():
    """ Test that B fails on SDs """
    d = SD([1/4]*4)
    assert_raises(ditException, B, d)

def test_R1():
    """ Test R for dependent variables """
    d = D(['00', '11'], [1/2, 1/2])
    assert_almost_equal(R(d), 0)

def test_R2():
    """ Test R for independent variables """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert_almost_equal(R(d), 2)
    assert_almost_equal(R(d, [[0], [1]]), 2)

def test_R3():
    """ Test R for a generic distribution """
    d = D(['000', '011', '101', '110'], [1/4]*4)
    assert_almost_equal(R(d), 0)
    assert_almost_equal(R(d, [[0, 1], [2]]), 1)

def test_R4():
    """ Test that R = H - I in two variables """
    d = D(['00', '01', '11'], [1/3]*3)
    assert_almost_equal(R(d), H(d)-I(d, [0], [1]))

def test_R5():
    """ Test that R fails on SDs """
    d = SD([1/4]*4)
    assert_raises(ditException, R, d)

def test_BR1():
    """ Test that B + R = H """
    d = D(['000', '001', '010', '100', '111'], [1/5]*5)
    assert_almost_equal(B(d)+R(d), H(d))
