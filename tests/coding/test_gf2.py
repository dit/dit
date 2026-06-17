"""
Tests for the GF(2) linear-algebra helpers.
"""

import numpy as np
import pytest

from dit.coding import _gf2
from dit.exceptions import ditException


def test_rank():
    """The rank counts the independent rows over GF(2)."""
    assert _gf2.rank([[1, 0], [0, 1]]) == 2
    assert _gf2.rank([[1, 1], [1, 1]]) == 1
    assert _gf2.rank([[0, 0], [0, 0]]) == 0


def test_nullspace_full_rank_is_empty():
    """A full-rank square matrix has a trivial null space."""
    basis = _gf2.nullspace([[1, 0], [0, 1]])
    assert basis.shape == (0, 2)


def test_nullspace_basis():
    """The null-space basis vectors are annihilated by the matrix."""
    H = np.array([[1, 1, 0], [0, 1, 1]])
    basis = _gf2.nullspace(H)
    for v in basis:
        assert not np.any(_gf2.matvec(H, v))


def test_inverse_roundtrip():
    """The GF(2) inverse satisfies A A^-1 = I."""
    A = np.array([[1, 1], [0, 1]])
    inv = _gf2.inverse(A)
    assert np.array_equal((A @ inv) % 2, np.eye(2, dtype=int))


def test_inverse_non_square_raises():
    """Inverting a non-square matrix raises."""
    with pytest.raises(ditException):
        _gf2.inverse([[1, 0, 1], [0, 1, 1]])


def test_inverse_singular_raises():
    """Inverting a singular matrix raises."""
    with pytest.raises(ditException):
        _gf2.inverse([[1, 1], [1, 1]])
