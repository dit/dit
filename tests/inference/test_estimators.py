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
    block_entropy,
    conditional_entropy_rate,
    entropy_0,
    entropy_1,
    entropy_2,
    entropy_from_counts,
    get_counts,
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


@pytest.mark.parametrize("estimator", sorted(ENTROPY_ESTIMATORS))
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
