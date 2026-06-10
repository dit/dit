"""
Tests for dit.inference.knn_estimators.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, lists

from dit.inference.knn_estimators import (
    _total_correlation_ksg_scipy,
    differential_entropy_knn,
    total_correlation_ksg,
)


@settings(max_examples=25)
@given(mean=floats(min_value=-2.0, max_value=2.0), std=floats(min_value=0.1, max_value=2.0))
def test_entropy_knn1(mean, std):
    """
    Test entropy of normal samples.
    """
    n = 100000
    data = np.random.normal(mean, std, n).reshape(n, 1)
    h = differential_entropy_knn(data)
    assert h == pytest.approx(np.log2(2 * np.pi * np.e * std**2) / 2, abs=1e-1)


@settings(max_examples=25)
@given(mean=floats(min_value=-2.0, max_value=2.0), std=floats(min_value=0.1, max_value=2.0))
def test_entropy_knn2(mean, std):
    """
    Test entropy of normal samples.
    """
    n = 100000
    data = np.random.normal(mean, std, n).reshape(n, 1)
    h = differential_entropy_knn(data, [0])
    assert h == pytest.approx(np.log2(2 * np.pi * np.e * std**2) / 2, abs=1e-1)


@settings(max_examples=5)
@given(
    means=lists(floats(min_value=-2.0, max_value=2.0), min_size=2, max_size=2),
    stds=lists(floats(min_value=0.1, max_value=2.0), min_size=2, max_size=2),
    rho=floats(min_value=-0.9, max_value=0.9),
)
def test_mi_knn1(means, stds, rho):
    """
    Test mutual information of multinormal samples.
    """
    cov = np.array([[stds[0] ** 2, stds[0] * stds[1] * rho], [stds[0] * stds[1] * rho, stds[1] ** 2]])
    n = 100000
    data = np.random.multivariate_normal(means, cov, n)
    mi = total_correlation_ksg(data, [[0], [1]])
    assert mi == pytest.approx(-np.log2(1 - rho**2) / 2, abs=1e-1)


@settings(max_examples=1)
@given(
    means=lists(floats(min_value=-2.0, max_value=2.0), min_size=3, max_size=3),
    stds=lists(floats(min_value=0.1, max_value=2.0), min_size=3, max_size=3),
    rho=floats(min_value=-0.9, max_value=0.9),
)
def test_cmi_knn1(means, stds, rho):
    """
    Test conditional mutual information of multinormal samples.
    """
    cov = np.array(
        [[stds[0] ** 2, stds[0] * stds[1] * rho, 0], [stds[0] * stds[1] * rho, stds[1] ** 2, 0], [0, 0, stds[2] ** 2]]
    )
    n = 150000
    data = np.random.multivariate_normal(means, cov, n)
    mi = total_correlation_ksg(data, [[0], [1]], [2])
    assert mi == pytest.approx(-np.log2(1 - rho**2) / 2, abs=1e-1)


def test_total_correlation_ksg_scipy_unconditioned():
    """The scipy backend recovers the MI of a correlated bivariate normal.

    ``total_correlation_ksg`` binds to the sklearn backend when it is
    installed, so exercise the scipy implementation directly.
    """
    rng = np.random.RandomState(0)
    rho = 0.6
    cov = [[1.0, rho], [rho, 1.0]]
    data = rng.multivariate_normal([0.0, 0.0], cov, 20000)
    mi = _total_correlation_ksg_scipy(data, [[0], [1]])
    assert mi == pytest.approx(-np.log2(1 - rho**2) / 2, abs=1e-1)


def test_total_correlation_ksg_scipy_conditioned():
    """The scipy backend runs the conditioned (crvs) code path."""
    rng = np.random.RandomState(0)
    rho = 0.6
    cov = [[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]]
    data = rng.multivariate_normal([0.0, 0.0, 0.0], cov, 20000)
    cmi = _total_correlation_ksg_scipy(data, [[0], [1]], [2])
    assert np.isfinite(cmi)
    assert cmi == pytest.approx(-np.log2(1 - rho**2) / 2, abs=1.5e-1)
