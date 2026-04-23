"""
Tests for the modified synergistic disclosure decomposition.

Gutknecht, Makkeh & Wibral (2023), "From Babel to Boole: The Logical
Organization of Information Decompositions", arXiv:2306.00734v2, eq. 52.

The key property MSD enforces (that vanilla SynDisc does not) is the PID
consistency condition: for every source collection a,
    I(a : T) = sum of atoms Pi(f) where f(a) = 1.
"""

import pytest

from dit.multivariate import coinformation
from dit.pid.distributions import bivariates
from dit.pid.syndisc import ModifiedSynDisc


def _source_mi(dist, source, target):
    """Mutual information I(source : target)."""
    return coinformation(dist, [list(source), list(target)])


def _is_accessible(node, source):
    """True if the atom at *node* contributes to I(source : T).

    An atom's parthood distribution has f(a)=1 (accessible from source *a*)
    iff *a* is not a subset of any constraint element in the node.
    """
    if not node:
        return True
    source_set = set(source)
    return not any(source_set <= set(c) for c in node)


# ─────────────────────────────────────────────────────────────────────────────
# Consistency condition (the raison d'etre of MSD)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("name", ["synergy", "redundant", "unique 1", "and"])
def test_consistency_condition(name):
    """For each individual source, accessible atoms sum to I(source : T)."""
    d = bivariates[name]
    msd = ModifiedSynDisc(d)

    for source in msd._sources:
        expected = _source_mi(d, source, msd._target)
        actual = sum(msd.get_atom(node) for node in msd._lattice if _is_accessible(node, source))
        assert actual == pytest.approx(expected, abs=1e-3), f"Consistency failed for {name!r}, source {source}"


# ─────────────────────────────────────────────────────────────────────────────
# Atom sum equals I(X; Y)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("name", ["synergy", "redundant", "unique 1", "and"])
def test_atom_sum_equals_mutual_info(name):
    d = bivariates[name]
    msd = ModifiedSynDisc(d)
    atom_sum = sum(msd.get_atom(node) for node in msd._lattice)
    assert atom_sum == pytest.approx(msd._total, abs=1e-4), f"Failed for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Known atom values: XOR (pure synergy, identical to vanilla)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_msd_xor():
    d = bivariates["synergy"]
    msd = ModifiedSynDisc(d)
    assert msd[((0,), (1,))] == pytest.approx(1.0, abs=1e-4)
    assert msd[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert msd[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert msd[()] == pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Known atom values: COPY (pure redundancy, identical to vanilla)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_msd_copy():
    d = bivariates["redundant"]
    msd = ModifiedSynDisc(d)
    assert msd[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)
    assert msd[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert msd[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert msd[()] == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Known atom values: AND (MSD differs from vanilla here)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_msd_and():
    d = bivariates["and"]
    msd = ModifiedSynDisc(d)
    assert msd[((0,), (1,))] == pytest.approx(0.3113, abs=1e-3)
    assert msd[((0,),)] == pytest.approx(0.1887, abs=1e-3)
    assert msd[((1,),)] == pytest.approx(0.1887, abs=1e-3)
    assert msd[()] == pytest.approx(0.1226, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone decomposition (uses multi-source nodes only, so unchanged)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("name", ["synergy", "redundant", "unique 1", "and"])
def test_backbone_nonnegative(name):
    d = bivariates[name]
    msd = ModifiedSynDisc(d)
    n = len(msd._sources)
    for m in range(1, n + 1):
        assert msd.get_backbone_atom(m) >= -1e-6, f"Failed for {name}, m={m}"


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("name", ["synergy", "redundant", "unique 1", "and"])
def test_backbone_sum_equals_mutual_info(name):
    d = bivariates[name]
    msd = ModifiedSynDisc(d)
    n = len(msd._sources)
    bb_sum = sum(msd.get_backbone_atom(m) for m in range(1, n + 1))
    assert bb_sum == pytest.approx(msd._total, abs=1e-4), f"Failed for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_to_string():
    d = bivariates["synergy"]
    msd = ModifiedSynDisc(d)
    s = msd.to_string()
    assert "S_msd" in s
