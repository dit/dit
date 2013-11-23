"""
Tests for dit.multivariate.total_correlation.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

from dit import Distribution as D, ScalarDistribution as SD
from dit.exceptions import ditException
from dit.multivariate import total_correlation as T
from dit.shannon import mutual_information as I
from dit.example_dists import n_mod_m

def test_tc1():
    """ Test T of parity distributions """
    for n in range(3, 7):
        d = n_mod_m(n, 2)
        yield assert_almost_equal, T(d), 1.0

def test_tc2():
    """ Test T of subvariables, with names """
    d = n_mod_m(4, 2)
    assert_almost_equal(T(d, [[0], [1], [2]]), 0.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X'], ['Y'], ['Z']]), 0.0)

def test_tc3():
    """ Test conditional T """
    d = n_mod_m(4, 2)
    assert_almost_equal(T(d, [[0], [1], [2]], [3]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X'], ['Y'], ['Z']], ['W']), 1.0)

def test_tc4():
    """ Test conditional T """
    d = n_mod_m(4, 2)
    assert_almost_equal(T(d, [[0], [1]], [2, 3]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X'], ['Y']], ['Z', 'W']), 1.0)

def test_tc5():
    """ Test T with subvariables """
    d = n_mod_m(4, 2)
    assert_almost_equal(T(d, [[0, 1], [2], [3]]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X', 'Y'], ['Z'], ['W']]), 1.0)

def test_tc6():
    """ Test T with overlapping subvariables """
    d = n_mod_m(4, 2)
    assert_almost_equal(T(d, [[0, 1, 2], [1, 2, 3]]), 3.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(T(d, [['X', 'Y', 'Z'], ['Y', 'Z', 'W']]), 3.0)

def test_tc7():
    """ Test T = I with two variables """
    outcomes = ['00', '01', '11']
    pmf = [1/3] * 3
    d = D(outcomes, pmf)
    assert_almost_equal(T(d), I(d, [0], [1]))
    d.set_rv_names("XY")
    assert_almost_equal(T(d), I(d, ['X'], ['Y']))

def test_tc8():
    """ Test that T fails on SDs """
    d = SD([1/3]*3)
    assert_raises(ditException, T, d)
