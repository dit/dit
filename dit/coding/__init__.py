"""
Coding: building source and channel codes.

The source-coding half constructs lossless codes for a source
:class:`~dit.Distribution` using a variety of classical algorithms, and exposes
the code-theoretic properties of the resulting codes (rate, redundancy,
efficiency, the Kraft sum, whether the code is prefix-free / uniquely decodable /
optimal, ...).

The channel-coding half builds binary (GF(2)) channel codes -- linear block codes
(repetition, parity-check, Hamming, Reed-Muller, Golay), LDPC, polar, and
convolutional codes -- and evaluates them against an arbitrary channel given as a
conditional :class:`~dit.Distribution` ``p(Y|X)`` (decoding, block-error
probability, and gap to capacity).
"""

from ._channel import binary_erasure_channel, binary_symmetric_channel
from .base import ChannelCoding, SourceCoding
from .block_codes import golay, hamming, parity_check, reed_muller, repetition
from .codes import (
    fano,
    golomb,
    huffman,
    length_limited_huffman,
    rice,
    shannon,
    shannon_fano_elias,
)
from .convolutional import ConvolutionalCode, convolutional
from .ldpc import LDPCCode, gallager, ldpc
from .linear import LinearCode
from .polar import PolarCode, polar
from .symbol_code import SymbolCode
from .tunstall import TunstallCode, tunstall
from .universal import (
    elias_delta,
    elias_gamma,
    elias_omega,
    fibonacci,
    unary,
    universal_code,
)

__all__ = (
    "ChannelCoding",
    "ConvolutionalCode",
    "LDPCCode",
    "LinearCode",
    "PolarCode",
    "SourceCoding",
    "SymbolCode",
    "TunstallCode",
    "binary_erasure_channel",
    "binary_symmetric_channel",
    "convolutional",
    "elias_delta",
    "elias_gamma",
    "elias_omega",
    "fano",
    "fibonacci",
    "gallager",
    "golay",
    "golomb",
    "hamming",
    "huffman",
    "ldpc",
    "length_limited_huffman",
    "parity_check",
    "polar",
    "reed_muller",
    "repetition",
    "rice",
    "shannon",
    "shannon_fano_elias",
    "tunstall",
    "unary",
    "universal_code",
)
