"""
Tests for dit.multivariate.coinformation.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

from dit import Distribution as D, ScalarDistribution as SD
from dit.multivariate import coinformation as I, entropy as H
from dit.exceptions import ditException
from dit.example_dists import n_mod_m

def test_coi1():
    """ Test I for xor """
    d = n_mod_m(3, 2)
    assert_almost_equal(I(d), -1.0)
    assert_almost_equal(I(d, [[0], [1], [2]]), -1.0)
    assert_almost_equal(I(d, [[0], [1], [2]], [2]), 0.0)
    assert_almost_equal(I(d, [[0], [1]], [2]), 1.0)

def test_coi2():
    """ Test I for larger parity distribution """
    d = n_mod_m(4, 2)
    assert_almost_equal(I(d), 1.0)

def test_coi3():
    """ Test I for subsets of variables, with and without names """
    d = n_mod_m(4, 2)
    assert_almost_equal(I(d, [[0], [1], [2]]), 0.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X'], ['Y'], ['Z']]), 0.0)

def test_coi4():
    """ Test conditional I, with and without names """
    d = n_mod_m(4, 2)
    assert_almost_equal(I(d, [[0], [1], [2]], [3]), -1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X'], ['Y'], ['Z']], ['W']), -1.0)

def test_coi5():
    """ Test conditional I, with and without names """
    d = n_mod_m(4, 2)
    assert_almost_equal(I(d, [[0], [1]], [2, 3]), 1.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X'], ['Y']], ['Z', 'W']), 1.0)

def test_coi6():
    """ Test conditional I, with and without names """
    d = n_mod_m(4, 2)
    assert_almost_equal(I(d, [[0]], [1, 2, 3]), 0.0)
    d.set_rv_names("XYZW")
    assert_almost_equal(I(d, [['X']], ['Y', 'Z', 'W']), 0.0)

def test_coi7():
    """ Test that H = I for one variable """
    outcomes = "ABC"
    pmf = [1/3]*3
    d = D(outcomes, pmf)
    assert_almost_equal(H(d), I(d))

def test_coi8():
    """ Test that I fails on ScalarDistributions """
    d = SD([1/3]*3)
    assert_raises(ditException, I, d)
