"""
Tests for NSB, Lempel-Ziv entropy rate, KSG local permutation, and FDR control.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from dit.inference import (
    benjamini_hochberg,
    conditional_mutual_information_test_knn,
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


def test_knn_local_permutation():
    rng = np.random.default_rng(3)
    n = 400
    z = rng.normal(size=n)
    x = z + 0.3 * rng.normal(size=n)
    y = z + 0.3 * rng.normal(size=n)
    data = np.stack([x, y, z], axis=1)
    marginal = conditional_mutual_information_test_knn(data, [[0], [1]], n_surrogates=19, prng=0)
    conditional = conditional_mutual_information_test_knn(data, [[0], [1]], [2], n_surrogates=19, prng=0)
    assert marginal.pvalue <= 0.05
    assert conditional.pvalue > 0.05
    w = x + 0.5 * y + 0.2 * rng.normal(size=n)
    dependent = conditional_mutual_information_test_knn(
        np.stack([w, y, z], axis=1), [[0], [1]], [2], n_surrogates=19, prng=0
    )
    assert dependent.pvalue <= 0.05


@pytest.mark.parametrize("dependent", [False, True])
def test_benjamini_hochberg_matches_scipy(dependent):
    p = np.random.default_rng(4).random(30) ** 3
    reject, adjusted = benjamini_hochberg(p, 0.1, dependent=dependent)
    expected = false_discovery_control(p, method="by" if dependent else "bh")
    assert np.allclose(adjusted, expected)
    assert np.array_equal(reject, expected <= 0.1)
    assert benjamini_hochberg([])[0].size == 0
