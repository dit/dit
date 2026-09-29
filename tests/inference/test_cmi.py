"""
Tests for the conditional mutual information estimate and the stationary bootstrap.
"""

import numpy as np
import pytest

from dit.inference import (
    conditional_mutual_information,
    dist_from_timeseries,
    stationary_bootstrap,
)
from dit.multivariate import coinformation


def _coupled(N, rng, coupling=0.9):
    """
    x is a sticky binary chain; y copies x's previous value with probability `coupling`.
    """
    x = np.zeros(N, dtype=int)
    y = np.zeros(N, dtype=int)
    for t in range(1, N):
        x[t] = x[t - 1] if rng.random() < 0.8 else 1 - x[t - 1]
        y[t] = x[t - 1] if rng.random() < coupling else rng.integers(2)
    return x, y


def test_cmi_independent_and_copy():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 5000)
    assert conditional_mutual_information(x, x) == pytest.approx(conditional_mutual_information(x, x, np.zeros_like(x)))
    assert conditional_mutual_information(x, rng.integers(0, 3, 5000), estimator="miller_madow") == pytest.approx(
        0, abs=5e-3
    )


def test_stationary_bootstrap_shape_and_blocks():
    data = np.arange(100)
    resamples = stationary_bootstrap(data, n=5, mean_block_length=10, prng=0)
    assert resamples.shape == (5, 100)
    steps = np.diff(resamples, axis=1) % 100
    assert np.mean(steps == 1) > 0.8
    multi = stationary_bootstrap(np.stack([data, -data], axis=1), n=2, prng=0)
    assert np.array_equal(multi[..., 0], -multi[..., 1])


def test_cmi_matches_plugin_distribution():
    """
    The lagged CMI I[y_t : x_{t-1} | y_{t-1}] equals the exact CMI of the plug-in distribution.
    """
    x, y = _coupled(2000, np.random.default_rng(0))
    d = dist_from_timeseries(np.stack([x, y], axis=1), history_length=1)
    # outcomes: (x_past, y_past, x_now, y_now)
    expected = coinformation(d, [[3], [0]], [1])
    assert conditional_mutual_information(y[1:], x[:-1], y[:-1]) == pytest.approx(expected)


def test_cmi_conditioning_removes_common_cause():
    rng = np.random.default_rng(2)
    z = rng.integers(0, 2, 5000)
    x = z ^ (rng.random(5000) < 0.1)
    y = z ^ (rng.random(5000) < 0.1)
    assert conditional_mutual_information(x, y) > 0.2
    assert conditional_mutual_information(x, y, z, estimator="miller_madow") == pytest.approx(0, abs=5e-3)
