from __future__ import division

from nose.tools import *

from dit import Distribution as D, ScalarDistribution as SD
from dit.exceptions import ditException
from dit.algorithms import (total_correlation as T,
                            mutual_information as I)

def test_tc1():
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4] * 4
    d = D(outcomes, pmf)
    assert_almost_equal(T(d), 1.0)

def test_tc2():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(T(d), 1.0)

def test_tc3():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(T(d, [[0], [1], [2]]), 0.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X'], ['Y'], ['Z']]), 0.0)

def test_tc4():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(T(d, [[0], [1], [2]], [3]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X'], ['Y'], ['Z']], ['W']), 1.0)

def test_tc5():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(T(d, [[0], [1]], [2, 3]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X'], ['Y']], ['Z', 'W']), 1.0)

def test_tc6():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(T(d, [[0, 1], [2], [3]]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X', 'Y'], ['Z'], ['W']]), 1.0)

def test_tc7():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(T(d, [[0, 1, 2], [1, 2, 3]]), 3.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X', 'Y', 'Z'], ['Y', 'Z', 'W']]), 3.0)

def test_tc8():
    outcomes = ['00', '01', '11']
    pmf = [1/3] * 3
    d = D(outcomes, pmf)
    assert_almost_equal(T(d), I(d, [0], [1]))
    d.set_rv_names("XY")
    assert_almost_equal(T(d), I(d, ['X'], ['Y']))

def test_tc9():
    d = SD([1/3]*3)
    assert_raises(ditException, T, d)
