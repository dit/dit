"""
Tests for dit.example_dists.n_mod_m.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

from numpy import log2

from dit.multivariate import interaction_information as II
from dit.example_dists import n_mod_m

def test_n_mod_m1():
    """ Test that n < 1 fails """
    assert_raises(ValueError, n_mod_m, -1, 1)

def test_n_mod_m2():
    """ Test that m < 1 fails """
    assert_raises(ValueError, n_mod_m, 1, -1)

def test_n_mod_m3():
    """ Test that noninteger n failes """
    assert_raises(ValueError, n_mod_m, 3/2, 1)

def test_n_mod_m4():
    """ Test that noninteger m fails """
    assert_raises(ValueError, n_mod_m, 1, 3/2)

def test_n_mod_m5():
    """ Test that the interaction information is always 1 """
    for n in range(3, 6):
        d = n_mod_m(n, 2)
        yield assert_almost_equal, II(d), 1.0

def test_n_mod_m6():
    """ Test that II is the log of m """
    for m in range(2, 5):
        d = n_mod_m(3, m)
        yield assert_almost_equal, II(d), log2(m)
