"""
Tests for the synergistic disclosure decomposition (Rosas et al. 2020).

Expected values from Table 1 of the paper (Section 3.2, bivariate n=2).
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.syndisc import SynDisc

# ─────────────────────────────────────────────────────────────────────────────
# Table 1: XOR (synergy)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_syndisc_xor():
    d = bivariates["synergy"]
    sd = SynDisc(d)
    assert sd[((0,), (1,))] == pytest.approx(1.0, abs=1e-4)
    assert sd[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert sd[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert sd[()] == pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: COPY (redundant)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_syndisc_copy():
    d = bivariates["redundant"]
    sd = SynDisc(d)
    assert sd[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)
    assert sd[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert sd[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert sd[()] == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Unique 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_syndisc_unique1():
    d = bivariates["unique 1"]
    sd = SynDisc(d)
    assert sd[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)
    # S^{0}_d = 1: constraint on X_0, unique info disclosed from X_1
    assert sd[((0,),)] == pytest.approx(1.0, abs=1e-4)
    assert sd[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert sd[()] == pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: AND
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_syndisc_and():
    d = bivariates["and"]
    sd = SynDisc(d)
    assert sd[((0,), (1,))] == pytest.approx(0.3113, abs=1e-3)
    assert sd[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert sd[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert sd[()] == pytest.approx(0.5, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Structural invariants
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_atom_sum_equals_mutual_info():
    for name in ["synergy", "redundant", "unique 1", "and"]:
        d = bivariates[name]
        sd = SynDisc(d)
        atom_sum = sum(sd.get_atom(node) for node in sd._lattice)
        assert atom_sum == pytest.approx(sd._total, abs=1e-4), f"Failed for {name}"


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_boundary_synergies():
    for name in ["synergy", "redundant", "unique 1", "and"]:
        d = bivariates[name]
        sd = SynDisc(d)
        assert sd.get_synergy(sd._lattice.top) == pytest.approx(sd._total, abs=1e-4)
        assert sd.get_synergy(sd._lattice.bottom) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergy_monotonicity():
    """S_alpha >= S_beta when alpha is a descendant of beta (Lemma 2)."""
    d = bivariates["and"]
    sd = SynDisc(d)
    for node in sd._lattice:
        for desc in sd._lattice.descendants(node):
            assert sd.get_synergy(desc) <= sd.get_synergy(node) + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Backbone decomposition
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_backbone_nonnegative():
    for name in ["synergy", "redundant", "unique 1", "and"]:
        d = bivariates[name]
        sd = SynDisc(d)
        n = len(sd._sources)
        for m in range(1, n + 1):
            assert sd.get_backbone_atom(m) >= -1e-6, f"Failed for {name}, m={m}"


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_backbone_sum_equals_mutual_info():
    for name in ["synergy", "redundant", "unique 1", "and"]:
        d = bivariates[name]
        sd = SynDisc(d)
        n = len(sd._sources)
        bb_sum = sum(sd.get_backbone_atom(m) for m in range(1, n + 1))
        assert bb_sum == pytest.approx(sd._total, abs=1e-4), f"Failed for {name}"


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_backbone_xor():
    d = bivariates["synergy"]
    sd = SynDisc(d)
    assert sd.get_backbone_atom(2) == pytest.approx(1.0, abs=1e-3)
    assert sd.get_backbone_atom(1) == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Self-synergy (TBC)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_self_synergy_tbc():
    """Two independent fair coins: self-synergy S_{01}(X->X) = 1 bit."""
    from dit.distconst import uniform

    d = uniform(["00", "01", "10", "11"])
    sd = SynDisc(d, sources=[[0], [1]], target=[0, 1])
    assert sd[((0,), (1,))] == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_to_string():
    d = bivariates["synergy"]
    sd = SynDisc(d)
    s = sd.to_string()
    assert "S_disc" in s
    assert "{0}{1}" in s


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_backbone_to_string():
    d = bivariates["synergy"]
    sd = SynDisc(d)
    s = sd.backbone_to_string()
    assert "backbone" in s
