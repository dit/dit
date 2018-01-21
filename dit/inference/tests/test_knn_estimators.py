"""
Tests for dit.inference.knn_estimators.
"""
from __future__ import division

from hypothesis import given, settings, unlimited
from hypothesis.strategies import floats, lists

import pytest

import numpy as np

from dit.inference.knn_estimators import differential_entropy_knn, total_correlation_ksg


@settings(deadline=None, timeout=unlimited, max_examples=25)
@given(mean=floats(min_value=-2.0, max_value=2.0),
       std=floats(min_value=0.1, max_value=2.0))
def test_entropy_knn1(mean, std):
    """
    Test entropy of normal samples.
    """
    n = 100000
    data = np.random.normal(mean, std, n).reshape(n, 1)
    h = differential_entropy_knn(data)
    assert h == pytest.approx(np.log2(2*np.pi*np.e*std**2)/2, abs=1e-1)


@settings(deadline=None, timeout=unlimited, max_examples=25)
@given(mean=floats(min_value=-2.0, max_value=2.0),
       std=floats(min_value=0.1, max_value=2.0))
def test_entropy_knn2(mean, std):
    """
    Test entropy of normal samples.
    """
    n = 100000
    data = np.random.normal(mean, std, n).reshape(n, 1)
    h = differential_entropy_knn(data, [0])
    assert h == pytest.approx(np.log2(2*np.pi*np.e*std**2)/2, abs=1e-1)


@settings(deadline=None, max_examples=5)
@given(means=lists(floats(min_value=-2.0, max_value=2.0), min_size=2, max_size=2),
       stds=lists(floats(min_value=0.1, max_value=2.0), min_size=2, max_size=2),
       rho=floats(min_value=-0.9, max_value=0.9))
def test_mi_knn1(means, stds, rho):
    """
    Test entropy of normal samples.
    """
    cov = np.array([[stds[0]**2, stds[0]*stds[1]*rho], [stds[0]*stds[1]*rho, stds[1]**2]])
    n = 100000
    data = np.random.multivariate_normal(means, cov, n)
    mi = total_correlation_ksg(data, [[0], [1]])
    assert mi == pytest.approx(-np.log2(1-rho**2)/2, abs=1e-1)


@settings(deadline=None, max_examples=1)
@given(means=lists(floats(min_value=-2.0, max_value=2.0), min_size=3, max_size=3),
       stds=lists(floats(min_value=0.1, max_value=2.0), min_size=3, max_size=3),
       rho=floats(min_value=-0.9, max_value=0.9))
def test_cmi_knn1(means, stds, rho):
    """
    Test entropy of normal samples.
    """
    cov = np.array([[stds[0]**2, stds[0]*stds[1]*rho, 0],
                    [stds[0]*stds[1]*rho, stds[1]**2, 0],
                    [0, 0, stds[2]**2]])
    n = 150000
    data = np.random.multivariate_normal(means, cov, n)
    mi = total_correlation_ksg(data, [[0], [1]], [2])
    assert mi == pytest.approx(-np.log2(1-rho**2)/2, abs=1e-1)
