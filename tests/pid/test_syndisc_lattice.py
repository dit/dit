"""
Tests for the constraint lattice construction used by the synergistic
disclosure decomposition.
"""

import pytest

from dit.pid.syndisc import (
    _build_constraint_lattice,
    _constraint_le,
    _transform_constraint,
)


# ─────────────────────────────────────────────────────────────────────────────
# n = 2
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def lattice_n2():
    sources = ((0,), (1,))
    return _transform_constraint(_build_constraint_lattice(sources))


def test_n2_node_count(lattice_n2):
    assert len(list(lattice_n2)) == 5


def test_n2_top_bottom(lattice_n2):
    assert lattice_n2.top == ()
    assert lattice_n2.bottom == ((0, 1),)


def test_n2_expected_nodes(lattice_n2):
    nodes = set(lattice_n2)
    expected = {(), ((0,),), ((1,),), ((0,), (1,)), ((0, 1),)}
    assert nodes == expected


def test_n2_descendants_top(lattice_n2):
    descs = set(lattice_n2.descendants(()))
    assert descs == {((0,),), ((1,),), ((0,), (1,)), ((0, 1),)}


def test_n2_descendants_bottom(lattice_n2):
    descs = set(lattice_n2.descendants(((0, 1),)))
    assert descs == set()


def test_n2_descendants_single(lattice_n2):
    descs = set(lattice_n2.descendants(((0,),)))
    assert ((0,), (1,)) in descs
    assert ((0, 1),) in descs


def test_n2_covers(lattice_n2):
    covers = set(lattice_n2.covers(((0,), (1,))))
    assert covers == {((0, 1),)}


def test_n2_pair_below_single(lattice_n2):
    """More constrained nodes are below less constrained ones."""
    assert lattice_n2._relationship(((0,), (1,)), ((0,),))


def test_n2_single_not_below_pair(lattice_n2):
    assert not lattice_n2._relationship(((0,),), ((0,), (1,)))


# ─────────────────────────────────────────────────────────────────────────────
# n = 3
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def lattice_n3():
    sources = ((0,), (1,), (2,))
    return _transform_constraint(_build_constraint_lattice(sources))


def test_n3_node_count(lattice_n3):
    assert len(list(lattice_n3)) == 19


def test_n3_top_bottom(lattice_n3):
    assert lattice_n3.top == ()
    assert lattice_n3.bottom == ((0, 1, 2),)


def test_n3_bottom_has_no_descendants(lattice_n3):
    assert len(list(lattice_n3.descendants(lattice_n3.bottom))) == 0


def test_n3_top_has_all_descendants(lattice_n3):
    descs = list(lattice_n3.descendants(lattice_n3.top))
    assert len(descs) == 18


# ─────────────────────────────────────────────────────────────────────────────
# Ordering predicate directly
# ─────────────────────────────────────────────────────────────────────────────


def test_constraint_le_empty_le_all():
    assert _constraint_le(frozenset(), frozenset({frozenset({0})}))
    assert _constraint_le(frozenset(), frozenset())


def test_constraint_le_nonempty_not_le_empty():
    assert not _constraint_le(frozenset({frozenset({0})}), frozenset())


def test_constraint_le_subset_relation():
    alpha = frozenset({frozenset({0})})
    beta = frozenset({frozenset({0, 1})})
    assert _constraint_le(alpha, beta)
    assert not _constraint_le(beta, alpha)


def test_constraint_le_antichain():
    alpha = frozenset({frozenset({0}), frozenset({1})})
    beta = frozenset({frozenset({0, 1})})
    assert _constraint_le(alpha, beta)
    assert not _constraint_le(beta, alpha)
