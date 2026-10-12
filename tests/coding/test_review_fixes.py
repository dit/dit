"""
Regression tests for coding bugs found in review.
"""

import numpy as np

from dit import Distribution
from dit.coding import convolutional, hamming, ldpc, polar_source, tunstall
from dit.example_channels import binary_symmetric_channel


def test_tunstall_single_symbol_terminates():
    code = tunstall(Distribution(["a"], [1.0]), 2)
    assert len(code.word_to_code) == 1


def test_golomb_parameter_is_optimal_for_geometric_source():
    from dit.coding.codes import _optimal_golomb_m

    theta = 0.8
    outcomes = list(range(60))
    probs = np.array([(1 - theta) * theta**n for n in outcomes])
    d = Distribution(outcomes, probs / probs.sum())
    # Gallager & van Voorhis: theta^m + theta^(m+1) <= 1 < theta^(m-1) + theta^m.
    m = _optimal_golomb_m(d)
    assert theta**m + theta ** (m + 1) <= 1 < theta ** (m - 1) + theta**m


def test_convolutional_k1_decodes():
    code = convolutional((1, 1), 4)
    message = (0, 1, 0, 1)
    assert list(code.decode(code.encode(message))) == list(message)


def test_ldpc_zero_iterations():
    code = ldpc(hamming(3).H, max_iterations=0)
    assert len(code.decode((0,) * 7, channel=binary_symmetric_channel(0.1))) == code.message_length


def test_polar_source_integer_alphabet_roundtrip():
    d = Distribution([(1,), (2,)], [0.9, 0.1])
    code = polar_source(d, 4)
    for block in [(1, 1, 2, 2), (2, 1, 2, 1)]:
        assert code.decode(code.encode(block)) == [b - 1 for b in block]


def test_polar_source_integer_side_information():
    d = Distribution([(0, 0), (1, 1)], [0.5, 0.5])
    code = polar_source(d, 4, rv=0, crvs=[1])
    block = (0, 1, 1, 0)
    assert code.decode(code.encode(block), side_information=list(block)) == list(block)
