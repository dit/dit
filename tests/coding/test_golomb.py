"""
Tests for dit.coding Golomb and Rice codes.
"""

import pytest

from dit import Distribution as D
from dit.coding import golomb, rice
from dit.exceptions import ditException


def geometric(theta, n):
    """A truncated geometric distribution over 0..n-1."""
    ps = [(1 - theta) * theta**k for k in range(n)]
    total = sum(ps)
    return D(list(range(n)), [p / total for p in ps])


def test_golomb_roundtrip():
    """Golomb codes recover the source sequence."""
    d = geometric(0.6, 12)
    code = golomb(d)
    outcomes = d.outcomes
    seq = [outcomes[i] for i in [0, 3, 1, 5, 2, 0, 8]]
    assert code.decode(code.encode(seq)) == seq


def test_golomb_prefix_free():
    """Golomb codes are prefix-free."""
    assert golomb(geometric(0.5, 16)).is_prefix_free()


def test_golomb_optimal_on_dyadic_geometric():
    """For theta=1/2 the optimal Golomb code achieves the entropy."""
    d = geometric(0.5, 16)
    code = golomb(d)
    # Equality holds for the untruncated geometric; truncation leaves a tiny gap.
    assert code.average_length() == pytest.approx(code.source_entropy(), abs=1e-3)


def test_golomb_m_one_is_unary():
    """Golomb with m=1 is the unary code on the integers."""
    d = geometric(0.3, 6)
    code = golomb(d, m=1)
    assert code.codebook[0] == "0"
    assert code.codebook[1] == "10"
    assert code.codebook[3] == "1110"


def test_rice_roundtrip():
    """Rice codes recover the source sequence."""
    d = geometric(0.7, 12)
    code = rice(d)
    outcomes = d.outcomes
    seq = [outcomes[i] for i in [0, 2, 4, 1, 6]]
    assert code.decode(code.encode(seq)) == seq


def test_rice_is_power_of_two_golomb():
    """Rice with parameter k equals Golomb with m=2**k."""
    d = geometric(0.6, 12)
    assert rice(d, k=2).codebook == golomb(d, m=4).codebook


def test_golomb_requires_integers():
    """Non-integer outcomes are rejected."""
    d = D(["a", "b"], [0.5, 0.5])
    with pytest.raises(ditException):
        golomb(d)


def test_golomb_binary_only():
    """Golomb coding is binary."""
    d = geometric(0.5, 8)
    with pytest.raises(ditException):
        golomb(d, radix=3)
