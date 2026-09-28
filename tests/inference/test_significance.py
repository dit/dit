"""
Tests for dit.inference.significance.
"""

import numpy as np
import pytest

from dit.inference import (
    bootstrap_ci,
    conditional_mutual_information,
    conditional_mutual_information_test,
    dist_from_timeseries,
    stationary_bootstrap,
    transfer_entropy,
    transfer_entropy_ci,
    transfer_entropy_test,
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


def test_transfer_entropy_matches_plugin_distribution():
    x, y = _coupled(2000, np.random.default_rng(0))
    d = dist_from_timeseries(np.stack([x, y], axis=1), history_length=1)
    # outcomes: (x_past, y_past, x_now, y_now)
    expected = coinformation(d, [[3], [0]], [1])
    assert transfer_entropy(x, y, 1) == pytest.approx(expected)


def test_cmi_independent_and_copy():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 5000)
    assert conditional_mutual_information(x, x) == pytest.approx(conditional_mutual_information(x, x, np.zeros_like(x)))
    assert conditional_mutual_information(x, rng.integers(0, 3, 5000), estimator="miller_madow") == pytest.approx(
        0, abs=5e-3
    )


def test_cmi_test_conditional_independence():
    """
    x and y share z but are conditionally independent.
    """
    rng = np.random.default_rng(2)
    z = rng.integers(0, 2, 1000)
    x = z ^ (rng.random(1000) < 0.1)
    y = z ^ (rng.random(1000) < 0.1)
    assert conditional_mutual_information_test(x, y, n_surrogates=99, prng=0).pvalue < 0.05
    assert conditional_mutual_information_test(x, y, z, n_surrogates=99, prng=0).pvalue > 0.05


@pytest.mark.parametrize("null", ["conditional", "whittle", "shift", "block"])
def test_transfer_entropy_test(null):
    x, y = _coupled(800, np.random.default_rng(3))
    forward = transfer_entropy_test(x, y, null=null, n_surrogates=99, prng=0)
    assert forward.pvalue < 0.05
    assert forward.null.shape == (99,)


def test_transfer_entropy_test_size_autocorrelated_source():
    """
    With an autocorrelated but uncoupled source, source-preserving nulls keep their size.
    """
    rng = np.random.default_rng(4)
    rejections = 0
    trials = 60
    for t in range(trials):
        x, _ = _coupled(300, rng)
        _, y = _coupled(300, rng)
        rejections += transfer_entropy_test(x, y, null="whittle", n_surrogates=49, prng=t).pvalue <= 0.05
    assert rejections / trials < 0.15


def test_stationary_bootstrap_shape_and_blocks():
    data = np.arange(100)
    resamples = stationary_bootstrap(data, n=5, mean_block_length=10, prng=0)
    assert resamples.shape == (5, 100)
    steps = np.diff(resamples, axis=1) % 100
    assert np.mean(steps == 1) > 0.8
    multi = stationary_bootstrap(np.stack([data, -data], axis=1), n=2, prng=0)
    assert np.array_equal(multi[..., 0], -multi[..., 1])


def test_transfer_entropy_ci_coverage():
    """
    The interval covers the large-sample transfer entropy at about the nominal rate.
    """
    rng = np.random.default_rng(5)
    truth = transfer_entropy(*_coupled(200000, rng))
    trials = 30
    covered = 0
    for t in range(trials):
        x, y = _coupled(1000, rng)
        low, high = transfer_entropy_ci(x, y, n_boot=100, mean_block_length=10, prng=t)
        covered += low < truth < high
    assert covered / trials > 0.75


def test_bootstrap_ci_long_blocks():
    """
    With blocks much longer than the history, the generic interval covers the estimate.
    """
    x, y = _coupled(1000, np.random.default_rng(6))
    te = transfer_entropy(x, y)
    for method in ("percentile", "basic"):
        low, high = bootstrap_ci(
            np.stack([x, y], axis=1),
            lambda d: transfer_entropy(d[:, 0], d[:, 1]),
            n_boot=100,
            mean_block_length=200,
            method=method,
            prng=0,
        )
        assert low < te < high


def test_invalid():
    with pytest.raises(ValueError):
        transfer_entropy([0, 1, 0], [0, 1, 0], source_history=0)
    with pytest.raises(ValueError):
        transfer_entropy([0, 1, 0], [0, 1, 0], lag=0)
    with pytest.raises(ValueError):
        transfer_entropy([0, 1, 0], [0, 1, 0], history_length="nope")
    with pytest.raises(ValueError):
        transfer_entropy_test([0, 1, 0, 1], [0, 1, 0, 1], null="nope", n_surrogates=2)
    with pytest.raises(ValueError):
        transfer_entropy([0, 1, 0], [0, 1])


def test_lagged_transfer_entropy():
    """
    y copies x from three steps back: only lag 3 (or a long enough source history) sees it.
    """
    rng = np.random.default_rng(10)
    x = rng.integers(0, 2, 5000)
    y = np.roll(x, 3)
    assert transfer_entropy(x, y, lag=3, estimator="miller_madow") == pytest.approx(1.0, abs=0.02)
    assert transfer_entropy(x, y, lag=1, estimator="miller_madow") == pytest.approx(0.0, abs=0.02)
    assert transfer_entropy(x, y, source_history=3, estimator="miller_madow") == pytest.approx(1.0, abs=0.02)


def test_zero_target_history_is_lagged_mutual_information():
    rng = np.random.default_rng(11)
    x = rng.integers(0, 3, 3000)
    y = np.roll(x, 1)
    expected = conditional_mutual_information(y[1:], x[:-1])
    assert transfer_entropy(x, y, history_length=0) == pytest.approx(expected)


def test_conditional_transfer_entropy_removes_common_driver():
    """
    z drives x (lag 1) and y (lag 2): x -> y looks informative unless z is conditioned on.
    """
    rng = np.random.default_rng(12)
    z = rng.integers(0, 2, 6000)
    flip = lambda p: (rng.random(len(z)) < p).astype(int)  # noqa: E731
    x = np.roll(z, 1) ^ flip(0.05)
    y = np.roll(z, 2) ^ flip(0.05)
    assert transfer_entropy(x, y) > 0.3
    assert transfer_entropy(x, y, conditions=[z], history_length=2) < 0.02
    result = transfer_entropy_test(x, y, conditions=[z], history_length=2, n_surrogates=49, prng=0)
    assert result.pvalue > 0.05


def test_auto_history_length():
    x, y = _coupled(3000, np.random.default_rng(13))
    assert transfer_entropy(x, y, history_length="auto", prng=0) == pytest.approx(transfer_entropy(x, y, 1), abs=0.05)
    low, high = transfer_entropy_ci(x, y, history_length="auto", n_boot=30, prng=0)
    assert low < high
