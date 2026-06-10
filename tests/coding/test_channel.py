"""
Tests for the dit.coding channel helpers.
"""

import numpy as np
import pytest

from dit.algorithms import channel_capacity
from dit.coding import binary_erasure_channel, binary_symmetric_channel
from dit.coding._channel import bhattacharyya, channel_arrays, log_likelihoods


def test_bsc_arrays():
    """The BSC transition matrix matches its definition."""
    inputs, outputs, P = channel_arrays(binary_symmetric_channel(0.1))
    assert list(inputs) == [0, 1]
    assert list(outputs) == [0, 1]
    assert pytest.approx(np.array([[0.9, 0.1], [0.1, 0.9]])) == P


def test_bec_arrays():
    """The BEC has a three-symbol output alphabet with an erasure column."""
    inputs, outputs, P = channel_arrays(binary_erasure_channel(0.2))
    assert list(inputs) == [0, 1]
    assert list(outputs) == [0, 1, 2]
    assert pytest.approx(np.array([[0.8, 0.0, 0.2], [0.0, 0.8, 0.2]])) == P


def test_bsc_capacity():
    """The BSC capacity matches 1 - H(p)."""
    p = 0.11
    cap, _ = channel_capacity(binary_symmetric_channel(p))
    expected = 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
    assert cap == pytest.approx(expected)


def test_bec_capacity():
    """The BEC capacity matches 1 - epsilon."""
    eps = 0.25
    cap, _ = channel_capacity(binary_erasure_channel(eps))
    assert cap == pytest.approx(1 - eps)


def test_log_likelihoods_symmetry():
    """The BSC log-likelihood ratios are symmetric and opposite-signed."""
    llr = log_likelihoods(binary_symmetric_channel(0.1))
    assert llr[0] == pytest.approx(-llr[1])
    assert llr[0] > 0


def test_bec_erasure_llr_zero():
    """An erasure carries no information (zero LLR)."""
    llr = log_likelihoods(binary_erasure_channel(0.2))
    assert llr[2] == pytest.approx(0.0)


def test_bhattacharyya_bsc():
    """The Bhattacharyya parameter of a BSC is 2 sqrt(p(1-p))."""
    p = 0.1
    assert bhattacharyya(binary_symmetric_channel(p)) == pytest.approx(2 * np.sqrt(p * (1 - p)))
