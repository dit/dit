"""
Tests for dit.inference.estimators.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import digamma

from dit.inference import (
    ENTROPY_ESTIMATORS,
    UndersamplingWarning,
    block_entropy,
    conditional_entropy_rate,
    entropy_0,
    entropy_1,
    entropy_2,
    entropy_from_counts,
    get_counts,
    jackknife_entropy,
)


def test_entropy_0_1():
    data = [0] * 7 + [1] * 3
    h0 = entropy_0(data, 1)
    assert h0 == pytest.approx(0.8812908992306927)


def test_entropy_0_2():
    data = [0] * 7 + [1] * 3
    h0 = entropy_0(data, 2)
    assert h0 == pytest.approx(1.2243944454059861)


def test_entropy_1_1():
    data = [0] * 7 + [1] * 3
    h1 = entropy_1(data, 1)
    assert h1 == pytest.approx(0.95790370730770369)


def test_entropy_1_2():
    data = [0] * 7 + [1] * 3
    h1 = entropy_1(data, 2)
    assert h1 == pytest.approx(1.4043376727383439)


def test_entropy_2_1():
    data = [0] * 7 + [1] * 3
    h2 = entropy_2(data, 1)
    assert h2 == pytest.approx(1.1187360918572902)


def test_entropy_2_2():
    data = [0] * 7 + [1] * 3
    h2 = entropy_2(data, 2)
    assert h2 == pytest.approx(1.3303313645210046)


@pytest.mark.parametrize("length", [1, 2])
def test_digamma_matches_entropy_1(length):
    data = [0] * 7 + [1] * 3
    counts = get_counts(data, length)
    assert entropy_from_counts(counts, "digamma") == pytest.approx(entropy_1(data, length))


def test_grassberger_vs_entropy_2():
    """
    entropy_2 is Grassberger (2003) with psi(N) in place of log(N).
    """
    data = [0] * 7 + [1] * 3
    counts = get_counts(data, 1)
    N = counts.sum()
    shift = (np.log(N) - digamma(N)) / np.log(2)
    assert entropy_from_counts(counts, "grassberger") == pytest.approx(entropy_2(data, 1) + shift)


def test_miller_madow():
    counts = np.array([7, 3])
    expected = entropy_from_counts(counts, "plugin") + 1 / (2 * 10) / np.log(2)
    assert entropy_from_counts(counts, "miller_madow") == pytest.approx(expected)


def test_chao_shen_no_singletons_close_to_plugin():
    counts = np.array([500, 500])
    assert entropy_from_counts(counts, "chao_shen") == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("estimator", sorted(set(ENTROPY_ESTIMATORS) - {"ansb", "bonachela"}))
def test_bias_reduction(estimator):
    """
    On undersampled uniform data, corrected estimators are no worse than plug-in.
    """
    rng = np.random.default_rng(0)
    K, N = 64, 64
    errors = {"plugin": [], estimator: []}
    for _ in range(50):
        counts = np.bincount(rng.integers(0, K, N), minlength=K)
        for name in errors:
            errors[name].append(entropy_from_counts(counts, name) - np.log2(K))
    assert abs(np.mean(errors[estimator])) <= abs(np.mean(errors["plugin"])) + 1e-12


def test_unknown_estimator():
    with pytest.raises(ValueError):
        entropy_from_counts([1, 2], "nope")
    with pytest.raises(ValueError):
        jackknife_entropy([1, 2], "nope")


# Reference values from infomeasure 0.6.3 (``im.entropy(data, approach=..., base=2)``)
# and, for Good-Turing, pyitlib 0.3 (``drv.entropy(data, estimator="GOOD-TURING", base=2)``).
_SMALL = [9, 1, 2, 2, 2, 9, 10, 6, 0, 1, 3, 5, 7, 5, 3, 1, 8, 8, 0, 1, 5, 4, 10, 6, 5] + [
    5,
    7,
    7,
    2,
    8,
    9,
    11,
    9,
    3,
    3,
    7,
    7,
    8,
    10,
    3,
    20,
    21,
]
_ZIPF_COUNTS = [134, 42, 16, 16, 8, 11, 5, 8, 2, 6, 4, 3, 1, 2, 2, 5, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 16]


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        ("zhang", 3.8348161653781214),
        ("chao_wang_jost", 3.9157634648542494),
        ("shrink", 3.8073549220576055),
        ("ansb", 5.83594184036061),
        ("bonachela", 4.0583224634276185),
        ("bayes_jeffreys", 3.6538031208612813),
        ("bayes_laplace", 3.6916823878601943),
        ("bayes_perks", 3.6035519398989733),
        ("bayes_minimax", 3.650256497264969),
    ],
)
def test_reference_values_small(estimator, expected):
    counts = np.unique(_SMALL, return_counts=True)[1]
    with pytest.warns(UndersamplingWarning):
        assert entropy_from_counts(counts, estimator) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        ("zhang", 3.244011042703),
        ("chao_wang_jost", 3.264529169247),
        ("shrink", 3.208663812431),
        ("bonachela", 3.579005229706),
        ("good_turing", 3.033194714935),
    ],
)
def test_reference_values_zipf(estimator, expected):
    assert entropy_from_counts(_ZIPF_COUNTS, estimator) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(("A", "N", "series"), [(0.5, 30, True), (0.05, 40, True), (0.3, 200, True), (1e-7, 50, False)])
def test_chao_wang_jost_tail(A, N, series):
    """
    Both forms of the Chao–Wang–Jost tail match a high-precision evaluation of the closed form.
    """
    mpmath = pytest.importorskip("mpmath")
    from dit.inference.estimators import _cwj_tail

    with mpmath.workdps(60):
        a, q = mpmath.mpf(A), 1 - mpmath.mpf(A)
        exact = q ** (1 - N) * (-mpmath.log(a) - mpmath.fsum(q**r / r for r in range(1, N)))
    assert _cwj_tail(A, N) == pytest.approx(float(exact), rel=1e-6)
    assert _cwj_tail(A, N, series=series) == pytest.approx(float(exact), rel=1e-6)


def test_chao_wang_jost_large_sample():
    counts = np.array([1] + [2] * 1000)
    h = entropy_from_counts(counts, "chao_wang_jost")
    assert np.isfinite(h)
    assert h >= entropy_from_counts(counts, "plugin")


def test_zhang_matches_definition():
    counts = np.array([5.0, 3, 1, 1])
    N = int(counts.sum())
    expected = 0.0
    for n in counts:
        inner = 0.0
        for v in range(1, N - int(n) + 1):
            inner += np.prod([1 + (1 - n) / (N - 1 - j) for j in range(v)]) / v
        expected += n / N * inner
    assert entropy_from_counts(counts, "zhang") == pytest.approx(expected / np.log(2))


def test_bayes_unseen_outcomes():
    """
    With an explicit alphabet the unseen outcomes receive pseudocounts.
    """
    counts = np.array([3, 1])
    total = 4 + 4 * 0.5
    p = np.array([3.5, 1.5, 0.5, 0.5]) / total
    expected = -np.sum(p * np.log2(p))
    assert entropy_from_counts(counts, "bayes_jeffreys", alphabet_size=4) == pytest.approx(expected)
    assert entropy_from_counts([3, 1, 0, 0], "bayes_jeffreys") == pytest.approx(expected)


def test_shrink_limits():
    assert entropy_from_counts([1, 1, 1, 1], "shrink") == pytest.approx(2.0)
    assert entropy_from_counts([5000, 5000], "shrink") == pytest.approx(1.0)
    assert entropy_from_counts([3, 1], "shrink", alphabet_size=8) <= 3.0


def test_ansb_undersampled():
    """
    ANSB beats plug-in when N is much smaller than K but coincidences occur.
    """
    rng = np.random.default_rng(4)
    K, N = 20000, 1000
    errors = {"plugin": [], "ansb": []}
    for _ in range(20):
        counts = np.unique(rng.integers(0, K, N), return_counts=True)[1]
        for name in errors:
            errors[name].append(entropy_from_counts(counts, name) - np.log2(K))
    assert abs(np.mean(errors["ansb"])) < abs(np.mean(errors["plugin"]))
    with pytest.raises(ValueError):
        entropy_from_counts([1, 1, 1], "ansb")


def test_bonachela_matches_definition():
    counts = np.array([4, 2, 1, 1])
    N = counts.sum()
    expected = sum((n + 1) * sum(1 / j for j in range(n + 2, N + 3)) for n in counts) / (N + 2)
    assert entropy_from_counts(counts, "bonachela") == pytest.approx(expected / np.log(2))


def test_good_turing_unseen_mass():
    counts = np.array([5, 3, 2, 2, 1, 1, 1])
    closed = entropy_from_counts(counts, "good_turing")
    opened = entropy_from_counts(counts, "good_turing", alphabet_size=20)
    assert opened > closed
    assert entropy_from_counts([4, 4], "good_turing") == pytest.approx(1.0)


def test_jackknife_plugin():
    counts = np.array([4.0, 2, 1])
    N = counts.sum()

    def H(c):
        c = c[c > 0]
        return entropy_from_counts(c, "plugin")

    loo = []
    for i, n in enumerate(counts):
        reduced = counts.copy()
        reduced[i] -= 1
        loo += [H(reduced)] * int(n)
    expected = N * H(counts) - (N - 1) * np.mean(loo)
    h, var = jackknife_entropy(counts)
    assert h == pytest.approx(expected)
    assert var == pytest.approx((N - 1) / N * np.sum((np.array(loo) - np.mean(loo)) ** 2))
    assert entropy_from_counts(counts, "jackknife") == pytest.approx(h)
    assert jackknife_entropy([3], "miller_madow") == (0.0, 0.0)


@settings(max_examples=30, deadline=None)
@given(
    st.sampled_from(sorted(set(ENTROPY_ESTIMATORS) - {"ansb"})),
    st.lists(st.floats(0.05, 1), min_size=2, max_size=6),
)
def test_consistency(estimator, weights):
    """
    Every estimator converges to the true entropy when N is much larger than K.
    """
    p = np.array(weights) / np.sum(weights)
    counts = np.round(p * 200_000)
    q = counts / counts.sum()
    truth = -np.sum(q * np.log2(q))
    assert entropy_from_counts(counts, estimator) == pytest.approx(truth, abs=1e-3)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(0, 3), min_size=5, max_size=80))
def test_conditional_entropy_rate_identities(seq):
    assert conditional_entropy_rate(seq, 0) == pytest.approx(block_entropy(seq, 1))
    assert conditional_entropy_rate(seq, 2) == pytest.approx(block_entropy(seq, 3) - block_entropy(seq, 2))
    assert block_entropy(seq, 1) == pytest.approx(entropy_0(seq, 1))


def test_conditional_entropy_rate_golden_mean():
    rng = np.random.default_rng(1)
    x, prev = [], 1
    for _ in range(20000):
        prev = 0 if prev == 1 else int(rng.integers(2))
        x.append(prev)
    for L in (1, 2, 3):
        assert conditional_entropy_rate(x, L, "grassberger") == pytest.approx(2 / 3, abs=0.02)
