"""
Tests for ordinal patterns, relative ranks, and permutation entropies.
"""

from itertools import permutations
from math import factorial, log2

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit.inference import (
    Trials,
    ordinal_patterns,
    permutation_entropy,
    relative_rank,
    weighted_permutation_entropy,
)

series = st.lists(st.integers(-1000, 1000).map(float), min_size=6, max_size=60)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_permutations_get_lexicographic_codes(order):
    for code, perm in enumerate(permutations(range(order))):
        assert ordinal_patterns(np.array(perm, dtype=float), order).tolist() == [code]


@settings(max_examples=50, deadline=None)
@given(ts=series, order=st.integers(2, 4), delay=st.integers(1, 2))
def test_monotone_invariance(ts, order, delay):
    x = np.array(ts)
    for transform in (np.arctan, lambda v: 3 * v + 7, lambda v: np.sign(v) * np.abs(v) ** 3):
        assert np.array_equal(ordinal_patterns(x, order, delay), ordinal_patterns(transform(x), order, delay))
        assert np.array_equal(relative_rank(x, order, delay), relative_rank(transform(x), order, delay))


@settings(max_examples=50, deadline=None)
@given(ts=series, order=st.integers(2, 4), delay=st.integers(1, 2))
def test_lengths_and_ranges(ts, order, delay):
    x = np.array(ts)
    patterns = ordinal_patterns(x, order, delay)
    ranks = relative_rank(x, order, delay)
    assert len(patterns) == max(len(x) - (order - 1) * delay, 0)
    assert len(ranks) == max(len(x) - order * delay, 0)
    assert patterns.min(initial=0) >= 0 and patterns.max(initial=0) < factorial(order)
    assert ranks.min(initial=0) >= 0 and ranks.max(initial=0) <= order


def test_relative_rank_alignment():
    x = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 5.0]
    assert relative_rank(x, 2).tolist() == [2, 2, 0, 0, 2]
    assert relative_rank(x, 2, ties="distinct").tolist() == [2, 1, 0, 0, 2]


def test_ties():
    x = [1.0, 1.0, 1.0, 2.0]
    assert ordinal_patterns(x, 3).tolist() == [0, 0]
    distinct = ordinal_patterns(x, 3, ties="distinct")
    assert distinct[0] != distinct[1]
    noisy = ordinal_patterns(x, 3, ties="noise", prng=0)
    assert noisy.max() < 6
    with pytest.raises(ValueError):
        ordinal_patterns(x, 3, ties="nope")


def test_permutation_entropy_extremes():
    rng = np.random.default_rng(0)
    assert permutation_entropy(np.arange(100.0), 3) == 0.0
    iid = rng.normal(size=20000)
    assert permutation_entropy(iid, 3) == pytest.approx(log2(6), abs=0.01)
    assert permutation_entropy(iid, 3, normalize=True) == pytest.approx(1.0, abs=0.01)


def test_logistic_map_forbidden_patterns():
    """
    The fully chaotic logistic map never produces the decreasing pattern (2, 1, 0).
    """
    x = [0.1234]
    for _ in range(5000):
        x.append(4 * x[-1] * (1 - x[-1]))
    patterns = set(ordinal_patterns(x, 3).tolist())
    assert 5 not in patterns
    assert permutation_entropy(x, 3) < log2(6)


def test_weighted_permutation_entropy():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=5000)
    assert weighted_permutation_entropy(iid, 3, normalize=True) == pytest.approx(1.0, abs=0.02)
    # Equal-variance windows make the weighted and plain entropies coincide.
    square = np.tile([0.0, 1.0], 500)
    assert weighted_permutation_entropy(square, 2) == pytest.approx(permutation_entropy(square, 2))
    # Large, structured oscillations dominate small noise in the weighted version.
    signal = np.sin(np.arange(4000) / 5) * 10 + rng.normal(scale=0.05, size=4000)
    assert weighted_permutation_entropy(signal, 3) < permutation_entropy(signal, 3)


def test_trials_pool():
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    pooled = permutation_entropy(Trials([a, b]), 3)
    assert pooled == pytest.approx(1.0)
    assert isinstance(ordinal_patterns(Trials([a, b]), 3), Trials)
    assert weighted_permutation_entropy(Trials([a, b]), 3) == pytest.approx(1.0)
