"""
Tests for dit.multivariate.coinformation.
"""

from __future__ import division

import pytest

from dit import Distribution as D, ScalarDistribution as SD
from dit.multivariate import coinformation as I, entropy as H
from dit.exceptions import ditException
from dit.example_dists import n_mod_m

def test_coi1():
    """ Test I for xor """
    d = n_mod_m(3, 2)
    assert I(d) == pytest.approx(-1.0)
    assert I(d, [[0], [1], [2]]) == pytest.approx(-1.0)
    assert I(d, [[0], [1], [2]], [2]) == pytest.approx(0.0)
    assert I(d, [[0], [1]], [2]) == pytest.approx(1.0)

def test_coi2():
    """ Test I for larger parity distribution """
    d = n_mod_m(4, 2)
    assert I(d) == pytest.approx(1.0)

def test_coi3():
    """ Test I for subsets of variables, with and without names """
    d = n_mod_m(4, 2)
    assert I(d, [[0], [1], [2]]) == pytest.approx(0.0)
    d.set_rv_names("XYZW")
    assert I(d, [['X'], ['Y'], ['Z']]) == pytest.approx(0.0)

def test_coi4():
    """ Test conditional I, with and without names """
    d = n_mod_m(4, 2)
    assert I(d, [[0], [1], [2]], [3]) == pytest.approx(-1.0)
    d.set_rv_names("XYZW")
    assert I(d, [['X'], ['Y'], ['Z']], ['W']) == pytest.approx(-1.0)

def test_coi5():
    """ Test conditional I, with and without names """
    d = n_mod_m(4, 2)
    assert I(d, [[0], [1]], [2, 3]) == pytest.approx(1.0)
    d.set_rv_names("XYZW")
    assert I(d, [['X'], ['Y']], ['Z', 'W']) == pytest.approx(1.0)

def test_coi6():
    """ Test conditional I, with and without names """
    d = n_mod_m(4, 2)
    assert I(d, [[0]], [1, 2, 3]) == pytest.approx(0.0)
    d.set_rv_names("XYZW")
    assert I(d, [['X']], ['Y', 'Z', 'W']) == pytest.approx(0.0)

def test_coi7():
    """ Test that H = I for one variable """
    outcomes = "ABC"
    pmf = [1/3]*3
    d = D(outcomes, pmf)
    assert H(d) == pytest.approx(I(d))

def test_coi8():
    """ Test that I fails on ScalarDistributions """
    d = SD([1/3]*3)
    with pytest.raises(ditException):
        I(d)
