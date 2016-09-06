"""
Tests for dit.multivariate.dual_total_correlation.
"""

from __future__ import division

import pytest

from dit import Distribution as D, ScalarDistribution as SD
from dit.multivariate import (dual_total_correlation as B,
                              residual_entropy as R)
from dit.shannon import (entropy as H,
                         mutual_information as I)
from dit.exceptions import ditException

def test_B1():
    """ Test B for two dependent variables """
    d = D(['00', '11'], [1/2, 1/2])
    assert B(d) == pytest.approx(1)

def test_B2():
    """ Test B for three dependent variables """
    d = D(['000', '111'], [1/2, 1/2])
    assert B(d) == pytest.approx(1)
    assert B(d, [[0], [1]], [2]) == pytest.approx(0)

def test_B3():
    """ Test B for four dependent variables """
    d = D(['0000', '1111'], [1/2, 1/2])
    assert B(d) == pytest.approx(1)
    assert B(d, [[0], [1, 2]], [3]) == pytest.approx(0)

def test_B4():
    """ Test B for xor distribution """
    d = D(['000', '011', '101', '110'], [1/4]*4)
    assert B(d) == pytest.approx(2)
    assert B(d, [[0], [1], [2]]) == pytest.approx(2)
    assert B(d, [[0], [1]], [2]) == pytest.approx(1)
    assert B(d, [[0], [2]], [1]) == pytest.approx(1)
    assert B(d, [[1], [2]], [0]) == pytest.approx(1)

def test_B5():
    """ Test B = I for two variables """
    d = D(['00', '01', '11'], [1/3]*3)
    assert B(d) == pytest.approx(I(d, [0], [1]))

def test_B6():
    """ Test that B fails on SDs """
    d = SD([1/4]*4)
    with pytest.raises(ditException):
        B(d)

def test_R1():
    """ Test R for dependent variables """
    d = D(['00', '11'], [1/2, 1/2])
    assert R(d) == pytest.approx(0)

def test_R2():
    """ Test R for independent variables """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert R(d) == pytest.approx(2)
    assert R(d, [[0], [1]]) == pytest.approx(2)

def test_R3():
    """ Test R for a generic distribution """
    d = D(['000', '011', '101', '110'], [1/4]*4)
    assert R(d) == pytest.approx(0)
    assert R(d, [[0, 1], [2]]) == pytest.approx(1)

def test_R4():
    """ Test that R = H - I in two variables """
    d = D(['00', '01', '11'], [1/3]*3)
    assert R(d) == pytest.approx(H(d)-I(d, [0], [1]))

def test_R5():
    """ Test that R fails on SDs """
    d = SD([1/4]*4)
    with pytest.raises(ditException):
        R(d)

def test_BR1():
    """ Test that B + R = H """
    d = D(['000', '001', '010', '100', '111'], [1/5]*5)
    assert B(d)+R(d) == pytest.approx(H(d))
