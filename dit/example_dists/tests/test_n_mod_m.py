from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

from numpy import log2

from dit.multivariate import interaction_information as II
from dit.example_dists import n_mod_m

def test_n_mod_m1():
    assert_raises(ValueError, n_mod_m, -1, 1)

def test_n_mod_m2():
    assert_raises(ValueError, n_mod_m, 1, -1)

def test_n_mod_m3():
    assert_raises(ValueError, n_mod_m, 3/2, 1)

def test_n_mod_m4():
    assert_raises(ValueError, n_mod_m, 1, 3/2)

def test_n_mod_m5():
    for i in range(3, 6):
        d = n_mod_m(i, 2)
        assert_almost_equal(II(d), 1.0)

def test_n_mod_m6():
    for i in range(2, 5):
        d = n_mod_m(3, i)
        assert_almost_equal(II(d), log2(i))
