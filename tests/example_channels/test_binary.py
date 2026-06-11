"""
Tests for the binary example channels.
"""

from math import log2

import numpy as np
import pytest

from dit.algorithms import channel_capacity
from dit.coding._channel import channel_arrays
from dit.example_channels import (
    binary_asymmetric_channel,
    binary_erasure_channel,
    binary_symmetric_channel,
    binary_symmetric_erasure_channel,
    z_channel,
)
from dit.exceptions import ditException


def _binary_entropy(p):
    if p in (0, 1):
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def test_bsc_is_conditional():
    """The BSC is a conditional distribution p(Y|X)."""
    assert binary_symmetric_channel(0.1).is_conditional()


def test_bsc_matrix():
    """The BSC transition matrix matches its definition."""
    inputs, outputs, P = channel_arrays(binary_symmetric_channel(0.1))
    assert list(inputs) == [0, 1]
    assert list(outputs) == [0, 1]
    assert np.allclose(P, [[0.9, 0.1], [0.1, 0.9]])


def test_bsc_capacity():
    """The BSC capacity is 1 - H(p)."""
    cap, _ = channel_capacity(binary_symmetric_channel(0.11))
    assert cap == pytest.approx(1 - _binary_entropy(0.11))


def test_bec_matrix_and_capacity():
    """The BEC has an erasure column and capacity 1 - epsilon."""
    inputs, outputs, P = channel_arrays(binary_erasure_channel(0.2))
    assert list(outputs) == [0, 1, 2]
    assert np.allclose(P, [[0.8, 0.0, 0.2], [0.0, 0.8, 0.2]])
    cap, _ = channel_capacity(binary_erasure_channel(0.25))
    assert cap == pytest.approx(0.75)


def test_z_channel_matrix():
    """The Z-channel passes 0 perfectly and flips 1 with probability p."""
    _, _, P = channel_arrays(z_channel(0.3))
    assert np.allclose(P, [[1.0, 0.0], [0.3, 0.7]])


def test_z_channel_capacity_bounds():
    """The Z-channel capacity lies strictly between the useless and noiseless limits."""
    cap, _ = channel_capacity(z_channel(0.3))
    assert 0 < cap < 1


def test_binary_asymmetric_matrix():
    """The binary asymmetric channel matches its two crossover probabilities."""
    _, _, P = channel_arrays(binary_asymmetric_channel(0.1, 0.2))
    assert np.allclose(P, [[0.9, 0.1], [0.2, 0.8]])


def test_bac_specializes_to_bsc():
    """A symmetric binary asymmetric channel equals the BSC."""
    _, _, P_bac = channel_arrays(binary_asymmetric_channel(0.15, 0.15))
    _, _, P_bsc = channel_arrays(binary_symmetric_channel(0.15))
    assert np.allclose(P_bac, P_bsc)


def test_bsec_matrix():
    """The error-and-erasure channel matches its definition."""
    _, outputs, P = channel_arrays(binary_symmetric_erasure_channel(0.1, 0.2))
    assert list(outputs) == [0, 1, 2]
    assert np.allclose(P, [[0.7, 0.1, 0.2], [0.1, 0.7, 0.2]])


def test_bsec_capacity_between_bsc_and_bec():
    """Adding erasures on top of errors lowers capacity below the pure BSC."""
    cap_bsec, _ = channel_capacity(binary_symmetric_erasure_channel(0.1, 0.2))
    cap_bsc, _ = channel_capacity(binary_symmetric_channel(0.1))
    assert 0 < cap_bsec < cap_bsc


def test_invalid_parameters_raise():
    """Out-of-range parameters raise."""
    with pytest.raises(ditException):
        binary_symmetric_channel(1.5)
    with pytest.raises(ditException):
        binary_symmetric_erasure_channel(0.6, 0.6)
