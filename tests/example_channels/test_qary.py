"""
Tests for the q-ary example channels.
"""

from math import log2

import numpy as np
import pytest

from dit.algorithms import channel_capacity
from dit.coding._channel import channel_arrays
from dit.example_channels import (
    binary_erasure_channel,
    binary_symmetric_channel,
    noisy_typewriter,
    q_ary_erasure_channel,
    q_ary_symmetric_channel,
)
from dit.exceptions import ditException


def _binary_entropy(p):
    if p in (0, 1):
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def test_qsc_row():
    """Each q-ary symmetric row keeps 1-p on the diagonal and spreads p evenly."""
    _, _, P = channel_arrays(q_ary_symmetric_channel(4, 0.3))
    assert np.allclose(P[0], [0.7, 0.1, 0.1, 0.1])


def test_qsc_capacity():
    """The q-ary symmetric capacity matches the closed form."""
    cap, _ = channel_capacity(q_ary_symmetric_channel(4, 0.3))
    expected = log2(4) - _binary_entropy(0.3) - 0.3 * log2(3)
    assert cap == pytest.approx(expected)


def test_qsc_specializes_to_bsc():
    """The q=2 symmetric channel equals the BSC."""
    _, _, P_q = channel_arrays(q_ary_symmetric_channel(2, 0.2))
    _, _, P_b = channel_arrays(binary_symmetric_channel(0.2))
    assert np.allclose(P_q, P_b)


def test_qec_outputs_and_capacity():
    """The q-ary erasure channel has an erasure symbol q and capacity (1-eps) log q."""
    _, outputs, P = channel_arrays(q_ary_erasure_channel(4, 0.25))
    assert list(outputs) == [0, 1, 2, 3, 4]
    assert np.allclose(P[0], [0.75, 0, 0, 0, 0.25])
    cap, _ = channel_capacity(q_ary_erasure_channel(4, 0.25))
    assert cap == pytest.approx(0.75 * log2(4))


def test_qec_specializes_to_bec():
    """The q=2 erasure channel equals the BEC."""
    _, _, P_q = channel_arrays(q_ary_erasure_channel(2, 0.2))
    _, _, P_b = channel_arrays(binary_erasure_channel(0.2))
    assert np.allclose(P_q, P_b)


def test_noisy_typewriter_matrix():
    """Each typewriter letter maps to itself or the next, each with probability 1/2."""
    _, _, P = channel_arrays(noisy_typewriter(4))
    assert np.allclose(
        P,
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5],
        ],
    )


def test_noisy_typewriter_capacity():
    """The noisy typewriter capacity is log2(n/2)."""
    cap, _ = channel_capacity(noisy_typewriter(6))
    assert cap == pytest.approx(log2(6 / 2))


def test_invalid_parameters_raise():
    """Bad alphabet sizes or probabilities raise."""
    with pytest.raises(ditException):
        q_ary_symmetric_channel(1, 0.1)
    with pytest.raises(ditException):
        q_ary_erasure_channel(4, 1.5)
    with pytest.raises(ditException):
        noisy_typewriter(1)
