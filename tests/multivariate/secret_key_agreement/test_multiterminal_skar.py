"""
Tests for dit.multivariate.secret_key_agreement.multiterminal_skar.
"""

import pytest
from hypothesis import given

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import caekl_mutual_information, entropy
from dit.multivariate.secret_key_agreement import omniscience_rate, secret_key_capacity
from dit.utils.testing import distributions

xor = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)


def test_xor_helper():
    """
    A helper holding the xor of two independent bits enables one secret bit.
    """
    rvs = [[0], [1], [2]]
    assert caekl_mutual_information(xor, [[0], [1]]) == pytest.approx(0.0)
    assert secret_key_capacity(xor, rvs, key_terminals=[0, 1]) == pytest.approx(1.0)
    assert omniscience_rate(xor, rvs, key_terminals=[0, 1]) == pytest.approx(1.0)


def test_xor_compromised():
    """
    Revealing the xor to the eavesdropper still leaves one private bit.
    """
    assert secret_key_capacity(xor, [[0], [1]], crvs=[2]) == pytest.approx(1.0)


def test_xor_all_terminals():
    """
    With every terminal in the key set, the capacity is the CAEKL value.
    """
    assert secret_key_capacity(xor) == pytest.approx(0.5)
    assert omniscience_rate(xor) == pytest.approx(1.5)


@pytest.mark.parametrize("key", [[0], [0, 3], [5, 1]])
def test_bad_key_terminals(key):
    """
    Key sets must contain at least two valid terminals.
    """
    with pytest.raises(ditException):
        secret_key_capacity(xor, key_terminals=key)


@given(dist=distributions(alphabets=(2,) * 4))
def test_caekl_equivalence(dist):
    """
    When all terminals share the key, the capacity equals CAEKL.
    """
    assert secret_key_capacity(dist) == pytest.approx(caekl_mutual_information(dist), abs=1e-6)
    assert secret_key_capacity(dist, crvs=[3], rvs=[[0], [1], [2]]) == pytest.approx(
        caekl_mutual_information(dist, rvs=[[0], [1], [2]], crvs=[3]), abs=1e-6
    )


@given(dist=distributions(alphabets=(2,) * 4))
def test_omniscience_complement(dist):
    """
    Capacity and omniscience rate sum to the joint entropy.
    """
    key = [0, 2]
    total = secret_key_capacity(dist, key_terminals=key) + omniscience_rate(dist, key_terminals=key)
    assert total == pytest.approx(entropy(dist), abs=1e-6)


@given(dist=distributions(alphabets=(2,) * 4))
def test_helpers_monotone(dist):
    """
    Adding helper terminals never decreases the capacity.
    """
    key = [0, 1]
    c2 = secret_key_capacity(dist, [[0], [1]])
    c3 = secret_key_capacity(dist, [[0], [1], [2]], key_terminals=key)
    c4 = secret_key_capacity(dist, [[0], [1], [2], [3]], key_terminals=key)
    assert c2 <= c3 + 1e-6
    assert c3 <= c4 + 1e-6


@given(dist=distributions(alphabets=(2,) * 4))
def test_key_set_monotone(dist):
    """
    Shrinking the key set never decreases the capacity.
    """
    assert secret_key_capacity(dist) <= secret_key_capacity(dist, key_terminals=[0, 1, 2]) + 1e-6
    assert secret_key_capacity(dist, key_terminals=[0, 1, 2]) <= secret_key_capacity(dist, key_terminals=[0, 1]) + 1e-6


@given(dist=distributions(alphabets=(2,) * 4))
def test_bounds(dist):
    """
    The capacity lies between zero and each key terminal's entropy.
    """
    key = [1, 3]
    c = secret_key_capacity(dist, key_terminals=key)
    assert c >= -1e-6
    assert c <= min(entropy(dist, [j]) for j in key) + 1e-6
