"""
Tests for dit.rate_distortion.curves
"""

import pytest

import numpy as np

from dit import Distribution
from dit.rate_distortion.curves import IBCurve, RDCurve
from dit.shannon import entropy


def test_simple_rd_1():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, beta_num=10)
    for r, d in zip(rd.rates, rd.distortions):
        assert r == pytest.approx(1 - entropy(d))


def test_simple_rd_2():
    """
    Test against know result, using blahut-arimoto.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, beta_num=10, method='ba')
    for r, d in zip(rd.rates, rd.distortions):
        assert r == pytest.approx(1 - entropy(d))


def test_simple_rd_3():
    """
    Test against know result, using blahut-arimoto.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, beta_num=10, beta_max=None, method='ba')
    for r, d in zip(rd.rates, rd.distortions):
        assert r == pytest.approx(1 - entropy(d))


def test_simple_ib_1():
    """
    Test against known values.
    """
    dist = Distribution(['00', '02', '12', '21', '22'], [1/5]*5)
    ib = IBCurve(dist, beta_max=10, beta_num=21)
    assert ib.complexities[2] == pytest.approx(0.0)
    assert ib.complexities[5] == pytest.approx(0.8)
    assert ib.complexities[20] == pytest.approx(1.5129028136502387)
    assert ib.relevances[2] == pytest.approx(0.0)
    assert ib.relevances[5] == pytest.approx(0.4)
    assert ib.relevances[20] == pytest.approx(0.5701613885745838)
    assert 3.0 in ib.find_kinks()