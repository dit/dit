"""
Tests for dit.rate_distortion.curves
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.rate_distortion.curves import IBCurve, RDCurve
from dit.rate_distortion.distortions import hamming, maximum_correlation, residual_entropy
from dit.shannon import entropy


@pytest.mark.flaky(reruns=5)
def test_simple_rd_1():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, beta_num=10)
    for r, d in zip(rd.rates, rd.distortions):
        assert r == pytest.approx(1 - entropy(d))


@pytest.mark.flaky(reruns=5)
def test_simple_rd_2():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, rv=[0], beta_num=10, alpha=0.5, distortion=maximum_correlation)
    assert rd.distortions[0] == pytest.approx(1.0)


@pytest.mark.flaky(reruns=5)
def test_simple_rd_3():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(['00', '01', '10', '11'], [1/4]*4)
    rd = RDCurve(dist, rv=[0], crvs=[1], beta_num=10, alpha=0.5, distortion=maximum_correlation)
    assert rd.distortions[0] == pytest.approx(1.0)


@pytest.mark.flaky(reruns=5)
def test_simple_rd_4():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, beta_num=10, alpha=0.0, distortion=residual_entropy)
    assert rd.distortions[0] == pytest.approx(1.0)


@pytest.mark.flaky(reruns=5)
def test_simple_rd_5():
    """
    Test against know result, using blahut-arimoto.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, beta_num=10, method='ba')
    for r, d in zip(rd.rates, rd.distortions):
        assert r == pytest.approx(1 - entropy(d))


@pytest.mark.flaky(reruns=5)
def test_simple_rd_6():
    """
    Test against know result, using blahut-arimoto.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RDCurve(dist, rv=[0], beta_num=10, beta_max=None, method='ba', distortion=residual_entropy)
    assert rd.distortions[0] == pytest.approx(2.0)


@pytest.mark.flaky(reruns=5)
def test_simple_ib_1():
    """
    Test against known values.
    """
    dist = Distribution(['00', '02', '12', '21', '22'], [1/5]*5)
    ib = IBCurve(dist, rvs=[[0], [1]], beta_max=10, beta_num=21)
    assert ib.complexities[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.complexities[5] == pytest.approx(0.8, abs=1e-4)
    assert ib.complexities[20] == pytest.approx(1.5129028136502387, abs=1e-4)
    assert ib.relevances[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.relevances[5] == pytest.approx(0.4, abs=1e-4)
    assert ib.relevances[20] == pytest.approx(0.5701613885745838, abs=1e-4)
    assert 3.0 in ib.find_kinks()


@pytest.mark.flaky(reruns=5)
def test_simple_ib_2():
    """
    Test against known values.
    """
    dist = Distribution(['00', '02', '12', '21', '22'], [1/5]*5)
    ib = IBCurve(dist, beta_max=None, beta_num=21, alpha=0.0)
    assert ib.complexities[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.complexities[12] == pytest.approx(0.97095059445466858, abs=1e-4)
    assert ib.complexities[20] == pytest.approx(1.5219280948873621, abs=1e-4)
    assert ib.relevances[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.relevances[12] == pytest.approx(0.4199730940219748, abs=1e-4)
    assert ib.relevances[20] == pytest.approx(0.5709505944546684, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_simple_ib_3():
    """
    Test against known values.
    """
    dist = Distribution(['00', '02', '12', '21', '22'], [1/5]*5)
    ib = IBCurve(dist, beta_max=None, beta_num=21, alpha=0.5)
    assert ib.complexities[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.complexities[5] == pytest.approx(0.8522009308325029, abs=1e-4)
    assert ib.complexities[20] == pytest.approx(1.5219280948873621, abs=1e-4)
    assert ib.relevances[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.relevances[5] == pytest.approx(0.4080081559717983, abs=1e-4)
    assert ib.relevances[20] == pytest.approx(0.5709505944546684, abs=1e-4)
