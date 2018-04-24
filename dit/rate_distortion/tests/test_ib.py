"""
Tests for dit.rate_distortion.information_bottleneck
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.divergences.pmf import variational_distance
from dit.exceptions import ditException
from dit.rate_distortion.information_bottleneck import InformationBottleneck, InformationBottleneckDivergence


dist = Distribution(['00', '02', '12', '21', '22'], [1/5]*5)
dist2 = Distribution(['000', '001', '020', '021', '120', '121', '210', '211', '220', '221'], [1/10]*10)


def test_ib_1():
    """
    Test simple IB.
    """
    ib = InformationBottleneck.functional()
    c, r = ib(dist, beta=0.0)
    assert c == pytest.approx(0.0)
    assert r == pytest.approx(0.0)


def test_ib_2():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, rvs=[[0]], beta=0.0)


def test_ib_3():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, beta=0.0, alpha=99)
        InformationBottleneck(dist, rvs=[[0]], beta=0.0)


def test_ib_4():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, beta=-10.0)


def test_ibd_1():
    """
    Test with custom distortion.
    """
    ibd = InformationBottleneckDivergence(dist, beta=0.0, divergence=variational_distance)
    ibd.optimize()
    pmf = ibd.construct_joint(ibd._optima)
    assert ibd.complexity(pmf) == pytest.approx(0.0)
    assert ibd.relevance(pmf) == pytest.approx(0.0)


def test_ibd_2():
    """
    Test with custom distortion.
    """
    ibd = InformationBottleneckDivergence(dist2, rvs=[[0], [1]], crvs=[2], beta=0.0, divergence=variational_distance)
    ibd.optimize()
    pmf = ibd.construct_joint(ibd._optima)
    assert ibd.complexity(pmf) == pytest.approx(0.0)
    assert ibd.relevance(pmf) == pytest.approx(0.0)