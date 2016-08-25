"""
Tests for dit.example_dists.n_mod_m.
"""

from __future__ import division
import pytest

from numpy import log2

from dit.multivariate import interaction_information as II
from dit.example_dists import n_mod_m

def test_n_mod_m1():
    """ Test that n < 1 fails """
    with pytest.raises(ValueError):
        n_mod_m(-1, 1)

def test_n_mod_m2():
    """ Test that m < 1 fails """
    with pytest.raises(ValueError):
        n_mod_m(1, -1)

def test_n_mod_m3():
    """ Test that noninteger n failes """
    with pytest.raises(ValueError):
        n_mod_m(3/2, 1)

def test_n_mod_m4():
    """ Test that noninteger m fails """
    with pytest.raises(ValueError):
        n_mod_m(1, 3/2)

@pytest.mark.parametrize('n', range(3, 6))
def test_n_mod_m5(n):
    """ Test that the interaction information is always 1 """
    d = n_mod_m(n, 2)
    assert II(d) == pytest.approx(1.0)

@pytest.mark.parametrize('m', range(2, 5))
def test_n_mod_m6(m):
    """ Test that II is the log of m """
    d = n_mod_m(3, m)
    assert II(d) == pytest.approx(log2(m))
