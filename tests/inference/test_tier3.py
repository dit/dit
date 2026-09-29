"""
Tests for NSB, Lempel-Ziv entropy rate, KSG local permutation, and FDR control.
"""

import warnings

import numpy as np
import pytest

from dit.inference import (
    entropy_from_counts,
    lz_entropy_rate,
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_nsb_less_biased_than_plugin_when_undersampled():
    rng = np.random.default_rng(0)
    K, N = 200, 100
    nsb, plugin = [], []
    for _ in range(20):
        counts = np.bincount(rng.integers(0, K, N), minlength=K)
        nsb.append(entropy_from_counts(counts, "nsb") - np.log2(K))
        plugin.append(entropy_from_counts(counts, "plugin") - np.log2(K))
    assert abs(np.mean(nsb)) < abs(np.mean(plugin)) / 4


def test_nsb_well_sampled_and_edge_cases():
    assert entropy_from_counts([5000, 5000], "nsb") == pytest.approx(1.0, abs=1e-3)
    assert entropy_from_counts([10], "nsb") == 0.0
    assert entropy_from_counts([4, 0, 0, 0], "nsb") > 0
    assert entropy_from_counts([3, 3], "nsb", alphabet_size=4) > entropy_from_counts([3, 3], "nsb")
    with pytest.raises(ValueError):
        entropy_from_counts([1, 1, 1], "nsb", alphabet_size=2)


def test_lz_entropy_rate():
    rng = np.random.default_rng(1)
    assert lz_entropy_rate(rng.integers(0, 2, 10000)) == pytest.approx(1.0, abs=0.15)
    assert lz_entropy_rate(rng.integers(0, 4, 10000)) == pytest.approx(2.0, abs=0.3)
    assert lz_entropy_rate((np.arange(3000) % 3 == 0).astype(int)) < 0.05
    with pytest.raises(ValueError):
        lz_entropy_rate([0])


def test_lz_golden_mean():
    rng = np.random.default_rng(2)
    x, prev = [], 1
    for _ in range(10000):
        prev = 0 if prev == 1 else int(rng.integers(2))
        x.append(prev)
    assert lz_entropy_rate(x) == pytest.approx(2 / 3, abs=0.1)
