"""
Tests for dit.algorithms.degrees_of_freedom.
"""

import pytest

import dit
from dit.algorithms import degrees_of_freedom


def test_dof_saturated():
    """The saturated model has df = (number of states) - 1."""
    d = dit.uniform_distribution(3, 2)
    assert degrees_of_freedom(d, [[0, 1, 2]]) == 7


def test_dof_independence():
    """The independence model has df = sum(cardinality_i - 1)."""
    d = dit.uniform_distribution(3, 2)
    assert degrees_of_freedom(d, [[0], [1], [2]]) == 3


def test_dof_independence_default():
    """The default structure is the independence model."""
    d = dit.uniform_distribution(3, 2)
    assert degrees_of_freedom(d) == 3


def test_dof_acyclic():
    """AB:BC has df = df(AB) + df(BC) - df(B) = 3 + 3 - 1 = 5."""
    d = dit.uniform_distribution(3, 2)
    assert degrees_of_freedom(d, [[0, 1], [1, 2]]) == 5


def test_dof_disjoint():
    """AB:C has df = df(AB) + df(C) = 3 + 1 = 4."""
    d = dit.uniform_distribution(3, 2)
    assert degrees_of_freedom(d, [[0, 1], [2]]) == 4


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        ([[0, 1, 2]], 7),
        ([[0, 1], [0, 2], [1, 2]], 6),
        ([[0, 1], [1, 2]], 5),
        ([[0, 1], [2]], 4),
        ([[0], [1], [2]], 3),
    ],
)
def test_dof_lattice_decreasing(structure, expected):
    """Degrees of freedom decrease down the structure lattice."""
    d = dit.uniform_distribution(3, 2)
    assert degrees_of_freedom(d, structure) == expected


def test_dof_mixed_cardinality():
    """df respects differing alphabet sizes: A in {0..3}, B in {0,1}."""
    d = dit.uniform_distribution(2, [[0, 1, 2, 3], [0, 1]])
    # saturated: 4*2 - 1 = 7
    assert degrees_of_freedom(d, [[0, 1]]) == 7
    # independence: (4-1) + (2-1) = 4
    assert degrees_of_freedom(d, [[0], [1]]) == 4
