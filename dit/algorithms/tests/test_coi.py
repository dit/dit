from __future__ import division

from nose.tools import *

from dit import Distribution as D, ScalarDistribution as SD
from dit.algorithms import coinformation as I, entropy as H
from dit.exceptions import ditException

def test_coi1():
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4] * 4
    d = D(outcomes, pmf)
    assert_almost_equal(I(d), -1.0)

def test_coi2():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(I(d), 1.0)

def test_coi3():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [[0],[1],[2]]), 0.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X'],['Y'],['Z']]), 0.0)

def test_coi4():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [[0],[1],[2]], [3]), -1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X'],['Y'],['Z']], ['W']), -1.0)

def test_coi5():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [[0],[1]], [2,3]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X'],['Y']], ['Z','W']), 1.0)

def test_coi6():
    outcomes = ['0000', '0011', '0101', '0110',
                '1001', '1010', '1100', '1111']
    pmf = [1/8] * 8
    d = D(outcomes, pmf)
    assert_almost_equal(I(d, [[0]], [1,2,3]), 0.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X']], ['Y','Z','W']), 0.0)

def test_coi7():
    outcomes = "ABC"
    pmf = [1/3]*3
    d = D(outcomes, pmf)
    assert_almost_equal(H(d), I(d))

def test_coi8():
    d = SD([1/3]*3)
    assert_raises(ditException, I, d)