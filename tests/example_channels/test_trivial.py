"""
Tests for the trivial endpoint channels and back-compatibility.
"""

from math import log2

import numpy as np
import pytest

from dit.algorithms import channel_capacity
from dit.coding._channel import channel_arrays
from dit.example_channels import identity_channel, useless_channel
from dit.exceptions import ditException


def test_identity_matrix_and_capacity():
    """The identity channel is the identity matrix with capacity log2(n)."""
    _, _, P = channel_arrays(identity_channel(4))
    assert np.allclose(P, np.eye(4))
    cap, _ = channel_capacity(identity_channel(4))
    assert cap == pytest.approx(log2(4))


def test_useless_matrix_and_capacity():
    """The useless channel has a uniform, input-independent output and zero capacity."""
    _, _, P = channel_arrays(useless_channel(3))
    assert np.allclose(P, np.full((3, 3), 1 / 3))
    cap, _ = channel_capacity(useless_channel(3))
    assert cap == pytest.approx(0.0, abs=1e-9)


def test_invalid_size_raises():
    """A non-positive alphabet size raises."""
    with pytest.raises(ditException):
        identity_channel(0)
    with pytest.raises(ditException):
        useless_channel(0)


def test_backwards_compatible_imports():
    """The BSC/BEC remain importable from dit.coding after relocation."""
    from dit.coding import binary_erasure_channel, binary_symmetric_channel
    from dit.example_channels import (
        binary_erasure_channel as bec,
    )
    from dit.example_channels import (
        binary_symmetric_channel as bsc,
    )

    assert binary_symmetric_channel is bsc
    assert binary_erasure_channel is bec
