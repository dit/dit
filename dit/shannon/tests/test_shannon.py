"""
Tests for dit.shannon.shannon.
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution as D, ScalarDistribution as SD
from dit.shannon import (entropy as H,
                         mutual_information as I,
                         conditional_entropy as CH,
                         entropy_pmf)

def test_entropy_pmf1d():
    """ Test the entropy of a fair coin """
    d = [.5, .5]
    assert entropy_pmf(d) == pytest.approx(1.0)

def test_entropy_pmf2d():
    """ Test the entropy of a fair coin """
    d = np.array([[1,0],[.5, .5]])
    H = np.array([0, 1])
    assert np.allclose(entropy_pmf(d), H)

def test_H1():
    """ Test the entropy of a fair coin """
    d = SD([1/2, 1/2])
    assert H(d) == pytest.approx(1.0)

def test_H2():
    """ Test the entropy of a fair coin, float style """
    assert H(1/2) == pytest.approx(1.0)

def test_H3():
    """ Test joint and marginal entropies """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert H(d, [0]) == pytest.approx(1.0)
    assert H(d, [1]) == pytest.approx(1.0)
    assert H(d, [0, 1]) == pytest.approx(2.0)
    assert H(d) == pytest.approx(2.0)

def test_H4():
    """ Test entropy in base 10 """
    d = SD([1/10]*10)
    d.set_base(10)
    assert H(d) == pytest.approx(1.0)

def test_I1():
    """ Test mutual information of independent variables """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert I(d, [0], [1]) == pytest.approx(0.0)

def test_I2():
    """ Test mutual information of dependent variables """
    outcomes = ['00', '11']
    pmf = [1/2]*2
    d = D(outcomes, pmf)
    assert I(d, [0], [1]) == pytest.approx(1.0)

def test_I3():
    """ Test mutual information of overlapping variables """
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert I(d, [0, 1], [1, 2]) == pytest.approx(2.0)

def test_CH1():
    """ Test conditional entropies """
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4]*4
    d = D(outcomes, pmf)
    assert CH(d, [0], [1, 2]) == pytest.approx(0.0)
    assert CH(d, [0, 1], [2]) == pytest.approx(1.0)
    assert CH(d, [0], [0]) == pytest.approx(0.0)
