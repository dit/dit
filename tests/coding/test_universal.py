"""
Tests for dit.coding universal integer codes.
"""

import pytest

from dit import Distribution as D
from dit.coding import (
    elias_delta,
    elias_gamma,
    elias_omega,
    fibonacci,
    unary,
    universal_code,
)
from dit.exceptions import ditException

KINDS = ["unary", "gamma", "delta", "omega", "fibonacci"]


def test_unary():
    assert unary(1) == "0"
    assert unary(2) == "10"
    assert unary(4) == "1110"


def test_elias_gamma():
    assert elias_gamma(1) == "1"
    assert elias_gamma(2) == "010"
    assert elias_gamma(3) == "011"
    assert elias_gamma(4) == "00100"


def test_elias_delta():
    assert elias_delta(1) == "1"
    assert elias_delta(2) == "0100"
    assert elias_delta(4) == "01100"


def test_elias_omega():
    assert elias_omega(1) == "0"
    assert elias_omega(2) == "100"
    assert elias_omega(4) == "101000"


def test_fibonacci():
    assert fibonacci(1) == "11"
    assert fibonacci(2) == "011"
    assert fibonacci(3) == "0011"
    assert fibonacci(4) == "1011"


@pytest.mark.parametrize("code", [unary, elias_gamma, elias_delta, elias_omega, fibonacci])
def test_integer_codes_positive_only(code):
    """Universal integer codes reject non-positive integers."""
    with pytest.raises(ditException):
        code(0)


@pytest.mark.parametrize("kind", KINDS)
def test_universal_code_roundtrip(kind):
    """Universal codes recover the source sequence and are prefix-free."""
    d = D(list(range(1, 9)), [1 / 8] * 8)
    code = universal_code(d, kind=kind)
    outcomes = d.outcomes
    seq = [outcomes[i] for i in [0, 4, 2, 7, 1]]
    assert code.is_prefix_free()
    assert code.decode(code.encode(seq)) == seq


def test_universal_code_unknown_kind():
    d = D(list(range(1, 4)), [1 / 3] * 3)
    with pytest.raises(ditException):
        universal_code(d, kind="nope")
