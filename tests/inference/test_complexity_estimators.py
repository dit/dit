"""
Tests for the Lempel-Ziv complexity, dispersion entropy, and multiscale
permutation entropy.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit.inference import dispersion_entropy, lz76_complexity, multiscale_permutation_entropy, permutation_entropy


def _series():
    """
    The series used to generate the antropy and pyEntrp references.
    """
    rng = np.random.default_rng(0)
    binary, quaternary = rng.integers(0, 2, 2000), rng.integers(0, 4, 500)
    walk = np.cumsum(rng.normal(size=3000))
    return binary, quaternary, walk


def test_lz76_kaspar_schuster_example():
    assert lz76_complexity(list("0001101001000101")) == 6
    assert lz76_complexity(list("0001101001000101"), normalize=True) == pytest.approx(1.5)


def test_lz76_matches_antropy():
    """
    Values from antropy's ``lziv_complexity``.
    """
    binary, quaternary, _ = _series()
    assert lz76_complexity(binary) == 188
    assert lz76_complexity(binary, normalize=True) == pytest.approx(1.0307837227582362)
    assert lz76_complexity(quaternary) == 112
    assert lz76_complexity(quaternary, normalize=True) == pytest.approx(1.0041678398821539)


def test_lz76_edge_cases():
    assert lz76_complexity([]) == 0
    assert lz76_complexity([0]) == 1
    assert lz76_complexity([1, 1, 1, 1]) == 2
    assert lz76_complexity([0, 1]) == 2
    periodic = [0, 1] * 500
    assert lz76_complexity(periodic) == 3


@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(0, 2), min_size=1, max_size=60))
def test_lz76_relabel_invariant(seq):
    relabeled = [{0: "a", 1: "b", 2: "c"}[s] for s in seq]
    assert lz76_complexity(seq) == lz76_complexity(relabeled)


def test_dispersion_entropy_matches_pyentrp():
    """
    Values from pyEntrp's ``dispersion_entropy``.
    """
    _, _, walk = _series()
    assert dispersion_entropy(walk, 3, 3, 1) == pytest.approx(1.8804435047454526)
    assert dispersion_entropy(walk, 6, 2, 2, mapping="linear", normalize=True) == pytest.approx(0.5607044610754359)
    assert dispersion_entropy(np.arange(1, 11), classes=3, order=2, normalize=True) == pytest.approx(0.667, abs=5e-4)


def test_dispersion_entropy_edge_cases():
    assert dispersion_entropy(np.ones(20)) == 0.0
    rng = np.random.default_rng(1)
    noise = rng.normal(size=20000)
    assert dispersion_entropy(noise, classes=2, order=2, normalize=True) == pytest.approx(1.0, abs=0.01)
    assert dispersion_entropy(noise, estimator="miller_madow") > dispersion_entropy(noise)
    with pytest.raises(ValueError):
        dispersion_entropy(noise, mapping="nope")
    with pytest.raises(ValueError):
        dispersion_entropy(noise[:3], order=3, delay=2)


def test_multiscale_permutation_entropy_matches_pyentrp():
    """
    Values from pyEntrp's ``multiscale_permutation_entropy(x, 3, 1, 4)``.
    """
    _, _, walk = _series()
    expected = [2.497833171225433, 2.42716196347381, 2.403897022784236, 2.406565919927721]
    assert multiscale_permutation_entropy(walk, 4) == pytest.approx(expected)
    assert multiscale_permutation_entropy(walk, 1)[0] == pytest.approx(permutation_entropy(walk))
