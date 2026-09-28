"""
Tests for dit.inference.posterior.
"""

import numpy as np
import pytest

from dit.inference import Trials, conditional_entropy_rate, entropy_rate_posterior


def _golden_mean(n, rng):
    x, prev = [], 1
    for _ in range(n):
        prev = 0 if prev == 1 else int(rng.integers(2))
        x.append(prev)
    return x


def test_posterior_covers_golden_mean():
    x = _golden_mean(3000, np.random.default_rng(0))
    post = entropy_rate_posterior(x, 1, n_samples=400, prng=1)
    low, high = post.interval(0.95)
    assert low < 2 / 3 < high
    assert post.mean == pytest.approx(conditional_entropy_rate(x, 1), abs=0.02)


def test_posterior_concentrates_with_data():
    rng = np.random.default_rng(2)
    small = entropy_rate_posterior(_golden_mean(200, rng), 1, n_samples=300, prng=0)
    large = entropy_rate_posterior(_golden_mean(20000, rng), 1, n_samples=300, prng=0)
    assert np.std(large.samples) < np.std(small.samples) / 3


def test_posterior_iid_coin_and_trials():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, 2000)
    post = entropy_rate_posterior(x, 0, n_samples=200, prng=0)
    assert post.interval(0.99)[1] <= 1.0 + 1e-12
    assert post.mean == pytest.approx(1.0, abs=0.01)
    trials = entropy_rate_posterior(Trials([x[:1000], x[1000:]]), 1, n_samples=50, prng=0)
    assert trials.order == 1


def test_posterior_invalid():
    with pytest.raises(ValueError):
        entropy_rate_posterior([0, 1], 1, prior=0)
    with pytest.raises(ValueError):
        entropy_rate_posterior(list(range(64)), 5)
