"""
Tests for dit.multivariate.entropy.
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution as D, ScalarDistribution as SD
from dit.example_dists import uniform
from dit.multivariate import entropy as H

def test_H1():
    """ Test H of a fair coin """
    d = D(['H', 'T'], [1/2, 1/2])
    assert H(d) == pytest.approx(1)

def test_H2():
    """ Test the various entropies of two independent events """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert H(d) == pytest.approx(2)
    assert H(d, [0]) == pytest.approx(1)
    assert H(d, [1]) == pytest.approx(1)
    assert H(d, [0], [1]) == pytest.approx(1)
    assert H(d, [1], [0]) == pytest.approx(1)
    assert H(d, [0], [0]) == pytest.approx(0)
    assert H(d, [0], [0, 1]) == pytest.approx(0)

def test_H3():
    """ Test the various entropies of two independent events with names """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    d.set_rv_names('XY')
    assert H(d) == pytest.approx(2)
    assert H(d, ['X']) == pytest.approx(1)
    assert H(d, ['Y']) == pytest.approx(1)
    assert H(d, ['X'], ['Y']) == pytest.approx(1)
    assert H(d, ['Y'], ['X']) == pytest.approx(1)
    assert H(d, ['X'], ['X']) == pytest.approx(0)
    assert H(d, ['X'], ['X', 'Y']) == pytest.approx(0)

@pytest.mark.parametrize('i', range(2, 10))
def test_H4(i):
    """ Test H for uniform distributions """
    d = D.from_distribution(uniform(i))
    assert H(d) == pytest.approx(np.log2(i))

@pytest.mark.parametrize('i', range(2, 10))
def test_H5(i):
    """ Test H for uniform distributions in various bases """
    d = D.from_distribution(uniform(i))
    d.set_base(i)
    assert H(d) == pytest.approx(1)

@pytest.mark.parametrize('i', range(2, 10))
def test_H6(i):
    """ Test H for uniform distributions using ScalarDistributions """
    d = uniform(i)
    assert H(d) == pytest.approx(np.log2(i))

@pytest.mark.parametrize('i', range(2, 10))
def test_H7(i):
    """ Test H for uniform distributions using SDs in various bases """
    d = uniform(i)
    d.set_base(i)
    assert H(d) == pytest.approx(1)
