"""
Tests for the optimization-based entropy estimators in dit.inference.
"""

from itertools import combinations

import numpy as np
import pytest

from dit.inference import entropy_from_counts
from dit.inference._advanced_estimators import (
    _best_approximation,
    _expected_subsample_plugin,
    _panzeri_treves_support,
    _pym,
)


def _uniform_counts(K, N, seed):
    rng = np.random.default_rng(seed)
    return np.bincount(rng.integers(0, K, N), minlength=K)


@pytest.mark.parametrize(
    ("degree", "expected"),
    [
        (1, [1.83939720588338906371e-01, -3.82732734394153339963e-12]),
        (2, [5.28191781375933477882e-02, 1.42697314194508684260e00, -1.53261149821644626634e00]),
        (
            3,
            [
                2.43505776253653272168e-02,
                2.23798522735207150447e00,
                -4.02124808854352533416e00,
                1.78326286118762600275e00,
            ],
        ),
    ],
)
def test_best_approximation(degree, expected):
    """
    Coefficients from Wu & Yang's reference implementation (``coeffs.txt``).
    """
    assert np.array(_best_approximation(degree)) == pytest.approx(expected, rel=1e-8, abs=1e-9)


@pytest.mark.parametrize(
    ("k", "fingerprint", "expected"),
    [
        (100_000, [(1, 2910), (2, 45)], 15.794990467501666),
        (1000, [(1, 300), (2, 60), (3, 10), (7, 2), (40, 1)], 9.6432929762),
    ],
)
def test_wu_yang_reference(k, fingerprint, expected):
    """
    Values from Wu & Yang's Python implementation (``Entropy(k).estimate(fin)``).
    """
    counts = np.concatenate([[f] * c for f, c in fingerprint])
    assert entropy_from_counts(counts, "wu_yang", alphabet_size=k) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("counts", "K", "expected"),
    [([3, 1, 1], 10, 5), ([5, 2, 1, 1, 1], 40, 8), ([2, 2, 1], 3, 3), ([1, 1, 1, 1], 100, 8)],
)
def test_panzeri_treves_support(counts, K, expected):
    """
    Values from pyentropy's ``pt_bayescount``.
    """
    counts = np.array(counts, dtype=float)
    assert _panzeri_treves_support(counts, counts.sum(), K) == expected


def test_panzeri_treves_correction():
    counts = np.array([3, 1, 1])
    plugin = entropy_from_counts(counts, "plugin")
    assert entropy_from_counts(counts, "panzeri_treves", alphabet_size=10) == pytest.approx(
        plugin + (5 - 1) / (2 * 5) / np.log(2)
    )


def test_expected_subsample_plugin_enumeration():
    """
    The hypergeometric expectation equals the average over every subsample.
    """
    counts = np.array([3, 2, 1])
    labels = np.repeat(np.arange(3), counts)
    for n in (2, 3):
        values = []
        for idx in combinations(range(len(labels)), n):
            p = np.bincount(labels[list(idx)]) / n
            p = p[p > 0]
            values.append(-np.sum(p * np.log(p)))
        assert _expected_subsample_plugin(counts.astype(float), 6, n) == pytest.approx(np.mean(values))


def test_quadratic_extrapolation_beats_plugin():
    K, N = 50, 60
    errors = {"plugin": [], "quadratic_extrapolation": []}
    for seed in range(20):
        counts = _uniform_counts(K, N, seed)
        for name in errors:
            errors[name].append(entropy_from_counts(counts, name) - np.log2(K))
    assert abs(np.mean(errors["quadratic_extrapolation"])) < 0.5 * abs(np.mean(errors["plugin"]))
    assert entropy_from_counts([2, 1], "quadratic_extrapolation") == entropy_from_counts([2, 1], "plugin")


def test_bub_k_max_one_is_miller_madow():
    """
    Paninski: with k_max = 1 BUB reduces to the Miller–Madow estimator.
    """
    from dit.inference._advanced_estimators import _bub

    counts = np.array([1.0, 2, 3, 4, 5, 4, 3, 2, 1])
    mm = entropy_from_counts(counts, "miller_madow") * np.log(2)
    assert _bub(counts, counts.sum(), 9, k_max=1) == pytest.approx(mm)


@pytest.mark.parametrize("estimator", ["wu_yang", "bub", "unseen"])
def test_undersampled_uniform(estimator):
    """
    On a uniform distribution over 1000 outcomes seen 400 times, plug-in is
    1.7 bits low; these estimators get within a quarter bit on average.
    """
    K = 1000
    errors = [
        entropy_from_counts(_uniform_counts(K, 400, seed), estimator, alphabet_size=K) - np.log2(K) for seed in range(5)
    ]
    assert abs(np.mean(errors)) < 0.25


def test_unseen_well_sampled_is_plugin():
    counts = np.array([500, 300, 200])
    assert entropy_from_counts(counts, "unseen") == pytest.approx(entropy_from_counts(counts, "plugin"))


def test_pym_pitman_yor():
    """
    On samples from Pitman–Yor distributions, PYM is nearly unbiased where plug-in is a bit low.
    """
    rng = np.random.default_rng(3)
    k = np.arange(1, 200_001)
    errors = {"pym": [], "plugin": []}
    for _ in range(5):
        v = rng.beta(1 - 0.6, 50 + k * 0.6)
        w = v * np.concatenate([[1], np.cumprod(1 - v)[:-1]])
        w /= w.sum()
        truth = -np.sum(w[w > 0] * np.log2(w[w > 0]))
        counts = np.bincount(rng.choice(len(w), 2000, p=w))
        counts = counts[counts > 0]
        for name in errors:
            errors[name].append(entropy_from_counts(counts, name) - truth)
    assert abs(np.mean(errors["pym"])) < 0.15
    assert np.mean(errors["plugin"]) < -0.5


def test_pym_quadrature_converges():
    counts = np.array([12, 7, 5, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    coarse, fine = (_pym(counts, counts.sum(), None, n_grid=g) for g in (25, 60))
    assert coarse == pytest.approx(fine, abs=1e-6)


def test_pym_errors():
    with pytest.raises(ValueError):
        entropy_from_counts([1, 1, 1], "pym")
    with pytest.raises(ValueError):
        entropy_from_counts([5], "pym")
