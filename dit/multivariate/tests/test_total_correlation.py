"""
Tests for dit.multivariate.total_correlation.
"""

from __future__ import division

import pytest

from dit import Distribution as D, ScalarDistribution as SD
from dit.exceptions import ditException
from dit.multivariate import total_correlation as T
from dit.shannon import mutual_information as I
from dit.example_dists import n_mod_m


@pytest.mark.parametrize('n', range(3, 7))
def test_tc1(n):
    """ Test T of parity distributions """
    d = n_mod_m(n, 2)
    assert T(d) == pytest.approx(1.0)

def test_tc2():
    """ Test T of subvariables, with names """
    d = n_mod_m(4, 2)
    assert T(d, [[0], [1], [2]]) == pytest.approx(0.0)
    d.set_rv_names("XYZW")
    assert T(d, [['X'], ['Y'], ['Z']]) == pytest.approx(0.0)

def test_tc3():
    """ Test conditional T """
    d = n_mod_m(4, 2)
    assert T(d, [[0], [1], [2]], [3]) == pytest.approx(1.0)
    d.set_rv_names("XYZW")
    assert T(d, [['X'], ['Y'], ['Z']], ['W']) == pytest.approx(1.0)

def test_tc4():
    """ Test conditional T """
    d = n_mod_m(4, 2)
    assert T(d, [[0], [1]], [2, 3]) == pytest.approx(1.0)
    d.set_rv_names("XYZW")
    assert T(d, [['X'], ['Y']], ['Z', 'W']) == pytest.approx(1.0)

def test_tc5():
    """ Test T with subvariables """
    d = n_mod_m(4, 2)
    assert T(d, [[0, 1], [2], [3]]) == pytest.approx(1.0)
    d.set_rv_names("XYZW")
    assert T(d, [['X', 'Y'], ['Z'], ['W']]) == pytest.approx(1.0)

def test_tc6():
    """ Test T with overlapping subvariables """
    d = n_mod_m(4, 2)
    assert T(d, [[0, 1, 2], [1, 2, 3]]) == pytest.approx(3.0)
    d.set_rv_names("XYZW")
    assert T(d, [['X', 'Y', 'Z'], ['Y', 'Z', 'W']]) == pytest.approx(3.0)

def test_tc7():
    """ Test T = I with two variables """
    outcomes = ['00', '01', '11']
    pmf = [1/3] * 3
    d = D(outcomes, pmf)
    assert T(d) == pytest.approx(I(d, [0], [1]))
    d.set_rv_names("XY")
    assert T(d) == pytest.approx(I(d, ['X'], ['Y']))

def test_tc8():
    """ Test that T fails on SDs """
    d = SD([1/3]*3)
    with pytest.raises(ditException):
        T(d)
