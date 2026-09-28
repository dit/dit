"""
Tests for dit.inference.surrogates.
"""

from collections import Counter
from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chisquare

from dit.inference import block_surrogates, shift_surrogates, whittle_count, whittle_surrogates


def _signature(seq, order):
    seq = list(seq)
    grams = Counter(tuple(seq[i : i + order + 1]) for i in range(len(seq) - order))
    return tuple(seq[:order]), grams


def _brute_force(seq, order):
    target = _signature(seq, order)
    return [s for s in product(sorted(set(seq)), repeat=len(seq)) if _signature(s, order) == target]


PETHEL = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1]


def test_pethel_example_count():
    """
    The worked example of Pethel & Hahs has 80 surrogates.
    """
    assert np.exp(whittle_count(PETHEL, 1)) == pytest.approx(80)


def test_pethel_example_uniform():
    """
    Surrogates are uniform over the 80 sequences.
    """
    surrogates = whittle_surrogates(PETHEL, 1, n=16000, prng=0)
    counts = Counter(map(tuple, surrogates.tolist()))
    assert set(counts) == set(_brute_force(PETHEL, 1))
    assert chisquare(list(counts.values())).pvalue > 1e-3


@settings(max_examples=40, deadline=None)
@given(
    seq=st.lists(st.integers(0, 2), min_size=3, max_size=9),
    order=st.integers(0, 2),
)
def test_whittle_count_brute_force(seq, order):
    """
    Whittle's formula counts the surrogate set exactly.
    """
    assert np.exp(whittle_count(seq, order)) == pytest.approx(len(_brute_force(seq, order)))


@settings(max_examples=40, deadline=None)
@given(
    seq=st.lists(st.integers(0, 3), min_size=1, max_size=60),
    order=st.integers(0, 3),
    seed=st.integers(0, 2**16),
)
def test_whittle_preserves_counts(seq, order, seed):
    """
    Every surrogate has the same (order + 1)-gram counts and initial word.
    """
    target = _signature(seq, order)
    for s in whittle_surrogates(seq, order, n=3, prng=seed):
        assert _signature(s.tolist(), order) == target


def test_whittle_joint_symbols():
    """
    Rows of 2D input are treated as joint symbols.
    """
    data = np.array([[0, 1], [1, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    surrogates = whittle_surrogates(data, 1, n=5, prng=1)
    assert surrogates.shape == (5, 6, 2)
    rows = {tuple(r) for r in data}
    assert all(tuple(r) in rows for s in surrogates for r in s)


def test_whittle_reproducible():
    x = [0, 1, 1, 0, 2, 2, 1, 0, 1, 2, 0, 0]
    assert np.array_equal(whittle_surrogates(x, 1, n=4, prng=7), whittle_surrogates(x, 1, n=4, prng=7))


def test_shift_surrogates():
    x = np.arange(10)
    for s in shift_surrogates(x, n=20, prng=0):
        assert s[0] != 0
        assert np.array_equal(np.sort(s), x)
        assert np.array_equal(np.roll(s, -s.tolist().index(0)), x)


def test_block_surrogates():
    x = np.arange(10)
    for s in block_surrogates(x, 3, n=10, prng=0):
        blocks = {tuple(s[i : i + 3]) for i in range(len(s))}
        assert {(0, 1, 2), (3, 4, 5), (6, 7, 8)} <= blocks | {(9,)}
        assert np.array_equal(np.sort(s), x)
