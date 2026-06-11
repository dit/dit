"""
Tests for dit.coding LDPC, polar, and convolutional codes.
"""

import itertools

import numpy as np
import pytest

from dit.coding import (
    LDPCCode,
    binary_erasure_channel,
    binary_symmetric_channel,
    convolutional,
    gallager,
    hamming,
    polar,
)
from dit.exceptions import ditException

# ── LDPC ─────────────────────────────────────────────────────────────────


def test_ldpc_noiseless_roundtrip():
    """Belief propagation recovers every codeword in the noiseless limit."""
    code = LDPCCode(hamming(3).H)
    channel = binary_symmetric_channel(0.01)
    for message in itertools.product((0, 1), repeat=code.message_length):
        codeword = code.encode(message)
        assert tuple(code.decode(codeword, channel=channel)) == message


def test_ldpc_error_rate_increases_with_noise():
    """A sparse Gallager code has a higher error rate at higher noise."""
    code = gallager(20, 3, 5, prng=np.random.default_rng(0))
    low = code.probability_of_error(
        binary_symmetric_channel(0.01),
        method="montecarlo",
        samples=1000,
        prng=np.random.default_rng(1),
    )
    high = code.probability_of_error(
        binary_symmetric_channel(0.1),
        method="montecarlo",
        samples=1000,
        prng=np.random.default_rng(2),
    )
    assert low < high


def test_gallager_shape():
    """The Gallager construction produces the expected parity-check shape."""
    code = gallager(12, 2, 4, prng=np.random.default_rng(0))
    assert code.H.shape == (6, 12)
    assert all(code.H.sum(axis=1) == 4)  # row weight wr
    assert all(code.H.sum(axis=0) == 2)  # column weight wc


def test_gallager_requires_divisibility():
    """An incompatible (n, wr) raises."""
    with pytest.raises(ditException):
        gallager(10, 2, 4)


# ── polar ────────────────────────────────────────────────────────────────


def test_polar_frozen_set_size():
    """A polar code freezes exactly n - k bit-channels."""
    code = polar(8, 4, binary_erasure_channel(0.3))
    assert int(code.frozen.sum()) == 8 - 4
    assert code.message_length == 4


def test_polar_noiseless_roundtrip():
    """Successive cancellation recovers messages over a near-perfect channel."""
    code = polar(8, 4, binary_erasure_channel(0.3))
    clean = binary_erasure_channel(1e-9)
    for message in itertools.product((0, 1), repeat=4):
        assert tuple(code.decode(code.encode(message), channel=clean)) == message


def test_polar_error_rate_increases_with_erasure():
    """The polar block-error probability grows with the erasure rate."""
    code = polar(8, 4, binary_erasure_channel(0.3))
    low = code.probability_of_error(
        binary_erasure_channel(0.1),
        method="montecarlo",
        samples=3000,
        prng=np.random.default_rng(0),
    )
    high = code.probability_of_error(
        binary_erasure_channel(0.5),
        method="montecarlo",
        samples=3000,
        prng=np.random.default_rng(1),
    )
    assert low < high


def test_polar_requires_power_of_two():
    """A non-power-of-two length raises."""
    with pytest.raises(ditException):
        polar(6, 3, binary_erasure_channel(0.3))


# ── convolutional ──────────────────────────────────────────────────────────


def test_convolutional_parameters():
    """The (7, 5) code is rate 1/2 with constraint length 3."""
    code = convolutional((0o7, 0o5), message_length=5)
    assert code.rate() == pytest.approx(0.5)
    assert code.K == 3


def test_convolutional_noiseless_roundtrip():
    """Hard-decision Viterbi recovers every terminated message."""
    code = convolutional((0o7, 0o5), message_length=5)
    for message in itertools.product((0, 1), repeat=5):
        assert code.decode(code.encode(message)) == list(message)


def test_convolutional_corrects_single_error():
    """The (7, 5) code corrects every single-bit error via Viterbi."""
    code = convolutional((0o7, 0o5), message_length=5)
    for message in itertools.product((0, 1), repeat=5):
        codeword = code.encode(message)
        for i in range(len(codeword)):
            received = list(codeword)
            received[i] ^= 1
            assert code.decode(received) == list(message)


def test_convolutional_soft_decoding():
    """Soft-decision Viterbi recovers messages over a clean BSC."""
    code = convolutional((0o7, 0o5), message_length=5)
    channel = binary_symmetric_channel(1e-6)
    for message in itertools.product((0, 1), repeat=5):
        assert code.decode(code.encode(message), channel=channel) == list(message)


def test_convolutional_requires_two_generators():
    """A single generator raises."""
    with pytest.raises(ditException):
        convolutional((0o7,), message_length=4)
