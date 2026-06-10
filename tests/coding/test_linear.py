"""
Tests for dit.coding linear block codes and the evaluation layer.
"""

import itertools

import numpy as np
import pytest

from dit.coding import (
    binary_symmetric_channel,
    golay,
    hamming,
    parity_check,
    reed_muller,
    repetition,
)
from dit.exceptions import ditException


def test_repetition_parameters():
    """The [n, 1, n] repetition code has the expected parameters."""
    code = repetition(5)
    assert code.length == 5
    assert code.dimension == 1
    assert code.minimum_distance() == 5
    assert code.rate() == pytest.approx(1 / 5)


def test_parity_check_parameters():
    """The single-parity-check code has minimum distance 2."""
    code = parity_check(4)
    assert code.length == 5
    assert code.dimension == 4
    assert code.minimum_distance() == 2


def test_hamming_parameters():
    """The Hamming [7, 4] code has minimum distance 3 and corrects one error."""
    code = hamming(3)
    assert (code.length, code.dimension) == (7, 4)
    assert code.minimum_distance() == 3
    assert code.error_correcting_capability() == 1


def test_hamming_corrects_every_single_error():
    """Hard-decision decoding corrects every single-bit error."""
    code = hamming(3)
    for message in itertools.product((0, 1), repeat=4):
        codeword = code.encode(message)
        for i in range(7):
            received = list(codeword)
            received[i] ^= 1
            assert tuple(code.decode(received)) == message


def test_reed_muller_parameters():
    """RM(1, 3) is the [8, 4, 4] first-order Reed-Muller code."""
    code = reed_muller(1, 3)
    assert (code.length, code.dimension) == (8, 4)
    assert code.minimum_distance() == 4


def test_golay_minimum_distances():
    """The Golay codes have the textbook minimum distances."""
    assert golay().minimum_distance() == 7
    assert golay(extended=True).minimum_distance() == 8


def test_encode_decode_roundtrip_noiseless():
    """Every message round-trips through encode/decode without noise."""
    code = hamming(3)
    for message in itertools.product((0, 1), repeat=4):
        assert tuple(code.decode(code.encode(message))) == message


def test_weight_enumerator():
    """The Hamming weight enumerator sums to the number of codewords."""
    code = hamming(3)
    enumerator = code.weight_enumerator()
    assert sum(enumerator.values()) == 2**4
    assert enumerator[0] == 1


def test_capacity_gap_positive_below_capacity():
    """A low-rate code over a clean-ish BSC operates below capacity."""
    code = repetition(3)
    gap = code.capacity_gap(binary_symmetric_channel(0.05))
    assert gap > 0


def test_probability_of_error_matches_closed_form():
    """Hamming over a BSC matches the closed-form block-error probability."""
    code = hamming(3)
    p = 0.05
    pe = code.probability_of_error(binary_symmetric_channel(p), method="exact")
    closed = 1 - (1 - p) ** 7 - 7 * p * (1 - p) ** 6
    assert pe == pytest.approx(closed)


def test_probability_of_error_montecarlo_close_to_exact():
    """Monte Carlo block-error probability tracks the exact value."""
    code = hamming(3)
    channel = binary_symmetric_channel(0.05)
    exact = code.probability_of_error(channel, method="exact")
    mc = code.probability_of_error(channel, method="montecarlo", samples=20000, prng=np.random.default_rng(0))
    assert mc == pytest.approx(exact, abs=0.02)


def test_intractable_enumeration_raises():
    """Enumerating codewords for a large code raises a clear error."""
    code = repetition(3)
    code.k = 25  # force the guard
    with pytest.raises(ditException):
        code.minimum_distance()
