"""
Edge-case and error-path coverage for the channel-coding modules.
"""

import numpy as np
import pytest

from dit.coding import (
    binary_erasure_channel,
    binary_symmetric_channel,
    convolutional,
    hamming,
    parity_check,
    reed_muller,
    repetition,
)
from dit.coding._channel import bhattacharyya, channel_arrays, log_likelihoods
from dit.coding.ldpc import LDPCCode
from dit.coding.linear import LinearCode
from dit.coding.polar import polar
from dit.example_channels import q_ary_symmetric_channel
from dit.exceptions import ditException

# ── LinearCode properties and guards ───────────────────────────────────────


def test_codewords_count_and_properties():
    """codewords() lists 2^k words and the matrix properties are exposed."""
    code = hamming(3)
    assert len(code.codewords()) == 2**code.dimension
    assert np.array_equal(code.generator_matrix, code.G)
    assert np.array_equal(code.parity_check_matrix, code.H)


def test_syndrome_decoding_intractable_raises():
    """Hard-decision decoding over too many syndromes raises."""
    code = repetition(22)  # n - k = 21 > the enumeration guard
    with pytest.raises(ditException):
        code.decode([0] * 22)


# ── block-code constructor validation ──────────────────────────────────────


def test_block_constructor_validation():
    """The block-code constructors reject degenerate parameters."""
    with pytest.raises(ditException):
        repetition(0)
    with pytest.raises(ditException):
        parity_check(0)
    with pytest.raises(ditException):
        hamming(1)
    with pytest.raises(ditException):
        reed_muller(2, 1)


# ── evaluation layer paths ─────────────────────────────────────────────────


def test_probability_of_error_auto_picks_montecarlo():
    """For a large output space the 'auto' method falls back to Monte Carlo."""
    code = repetition(21)  # 2^1 * 2^21 exceeds the exact-enumeration threshold
    pe = code.probability_of_error(binary_symmetric_channel(0.4), method="auto", samples=50)
    assert 0.0 <= pe <= 1.0


def test_probability_of_error_exact_over_erasure_channel():
    """Exact enumeration skips the zero-probability received words of a BEC."""
    code = hamming(3)
    pe = code.probability_of_error(binary_erasure_channel(0.1), method="exact")
    assert 0.0 <= pe < 1.0


# ── channel helper guards ──────────────────────────────────────────────────


def test_channel_arrays_requires_conditional():
    """A non-conditional distribution is rejected."""
    import dit

    joint = dit.Distribution(["00", "11"], [0.5, 0.5])
    with pytest.raises(ditException):
        channel_arrays(joint)


def test_soft_helpers_require_binary_input():
    """The LLR and Bhattacharyya helpers reject non-binary-input channels."""
    qary = q_ary_symmetric_channel(3, 0.1)
    with pytest.raises(ditException):
        log_likelihoods(qary)
    with pytest.raises(ditException):
        bhattacharyya(qary)


# ── LDPC guards and hard decoding ──────────────────────────────────────────


def test_ldpc_no_information_bits_raises():
    """A full-rank parity-check matrix leaves no information bits."""
    with pytest.raises(ditException):
        LDPCCode(np.eye(3, dtype=int))


def test_ldpc_hard_syndrome_decode():
    """Without a channel, an LDPC code falls back to hard syndrome decoding."""
    code = LDPCCode(hamming(3).H)
    for message in [(0, 0, 0, 0), (1, 0, 1, 1)]:
        codeword = list(code.encode(message))
        codeword[0] ^= 1  # a single, correctable error
        assert tuple(code.decode(codeword)) == message


def test_ldpc_constructor_and_gallager_guards():
    """The ldpc() constructor builds a code and gallager() validates its weights."""
    from dit.coding import gallager, ldpc

    code = ldpc(hamming(3).H)
    assert code.n == 7
    with pytest.raises(ditException):
        gallager(12, 4, 4)  # requires wc < wr
    # A default PRNG path (prng=None).
    assert gallager(12, 2, 4).H.shape == (6, 12)


# ── polar guards ───────────────────────────────────────────────────────────


def test_polar_dimension_validation():
    """An out-of-range polar dimension raises."""
    channel = binary_erasure_channel(0.3)
    with pytest.raises(ditException):
        polar(8, 9, channel)
    with pytest.raises(ditException):
        polar(8, 0, channel)


def test_polar_decode_requires_channel():
    """Polar decoding with no available channel raises."""
    code = polar(8, 4, binary_erasure_channel(0.3))
    code.channel = None
    with pytest.raises(ditException):
        code.decode([0] * 8, channel=None)


# ── convolutional ──────────────────────────────────────────────────────────


def test_convolutional_message_length():
    """The convolutional message length is exposed."""
    code = convolutional((0o7, 0o5), message_length=5)
    assert code.message_length == 5


def test_linear_code_message_length_matches_dimension():
    """A linear code's message length equals its dimension."""
    code = LinearCode([[1, 0, 1], [0, 1, 1]])
    assert code.message_length == code.dimension == 2
