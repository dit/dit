"""
Tests for the standalone synergistic disclosure scalar functions.
"""

import pytest

from dit.distconst import uniform
from dit.multivariate.synergistic_disclosure import (
    backbone_disclosure,
    modified_synergistic_disclosure,
    self_synergy,
    synergistic_disclosure,
)
from dit.pid import syndisc as syndisc_module
from dit.pid.distributions import bivariates


def _stub_optimize(monkeypatch):
    """Replace the (slow) basin-hopping optimize with a no-op that records a
    valid optimization vector, so the surrounding wrapper logic still runs."""

    def fake_optimize(self, niter=None, **kwargs):
        self._optima = self.construct_random_initial()

    monkeypatch.setattr(syndisc_module.SyndiscOptimizer, "optimize", fake_optimize)


# ─────────────────────────────────────────────────────────────────────────────
# synergistic_disclosure
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.flaky(reruns=5)
def test_synergistic_disclosure_xor():
    d = bivariates["synergy"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val == pytest.approx(1.0, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_synergistic_disclosure_copy():
    d = bivariates["redundant"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val == pytest.approx(0.0, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_synergistic_disclosure_empty_alpha():
    d = bivariates["synergy"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[])
    assert val == pytest.approx(1.0, abs=1e-4)


def test_synergistic_disclosure_empty_alpha_fast():
    """Empty alpha takes the optimizer-free coinformation fallback."""
    d = bivariates["redundant"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[])
    assert val == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# modified_synergistic_disclosure
# ─────────────────────────────────────────────────────────────────────────────


def test_modified_synergistic_disclosure_empty_alpha():
    """Empty alpha falls back to the full source-target coinformation."""
    d = bivariates["synergy"]
    val = modified_synergistic_disclosure(d, [[0], [1]], [2], alpha=[])
    assert val == pytest.approx(1.0)


def test_modified_synergistic_disclosure_singleton_alpha_synergy():
    """Singleton alpha uses I(T:all) - I(T:alpha); XOR leaks all 1 bit."""
    d = bivariates["synergy"]
    val = modified_synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0]])
    assert val == pytest.approx(1.0)


def test_modified_synergistic_disclosure_singleton_alpha_redundant():
    """A constrained source carrying all the target info leaves nothing."""
    d = bivariates["redundant"]
    val = modified_synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0]])
    assert val == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# backbone_disclosure
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.flaky(reruns=5)
def test_backbone_disclosure_xor():
    d = bivariates["synergy"]
    bb = backbone_disclosure(d)
    assert bb[2] == pytest.approx(1.0, abs=1e-3)
    assert bb[1] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_backbone_disclosure_copy():
    d = bivariates["redundant"]
    bb = backbone_disclosure(d)
    assert bb[1] == pytest.approx(1.0, abs=1e-3)
    assert bb[2] == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# self_synergy
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.flaky(reruns=5)
def test_self_synergy_two_coins():
    d = uniform(["00", "01", "10", "11"])
    val = self_synergy(d, sources=[[0], [1]])
    assert val == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests (optimizer stubbed -- exercise wrapper logic without optimizing)
# ─────────────────────────────────────────────────────────────────────────────


def test_synergistic_disclosure_smoke(monkeypatch):
    """The optimizer-backed path runs and clamps to a non-negative value."""
    _stub_optimize(monkeypatch)
    d = bivariates["synergy"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val >= 0.0


def test_synergistic_disclosure_exception_returns_zero(monkeypatch):
    """A failure during optimization is swallowed and yields 0.0."""

    def boom(self, niter=None, **kwargs):
        raise RuntimeError("optimization failed")

    monkeypatch.setattr(syndisc_module.SyndiscOptimizer, "optimize", boom)
    d = bivariates["synergy"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val == 0.0


def test_modified_synergistic_disclosure_multi_alpha_smoke(monkeypatch):
    """A multi-element alpha falls through to the optimizer-backed path."""
    _stub_optimize(monkeypatch)
    d = bivariates["synergy"]
    val = modified_synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val >= 0.0


def test_self_synergy_smoke(monkeypatch):
    """Self-synergy runs end-to-end with the optimizer stubbed."""
    _stub_optimize(monkeypatch)
    d = uniform(["00", "01", "10", "11"])
    val = self_synergy(d, sources=[[0], [1]])
    assert val >= 0.0


def test_self_synergy_default_sources_smoke(monkeypatch):
    """sources=None uses dist.rvs and alpha=None builds singleton constraints."""
    _stub_optimize(monkeypatch)
    d = uniform(["00", "01", "10", "11"])
    val = self_synergy(d)
    assert val >= 0.0


def test_backbone_disclosure_smoke(monkeypatch):
    """The backbone decomposition runs with the per-node solve stubbed out."""
    monkeypatch.setattr(syndisc_module.SynDisc, "_compute_s_alpha", lambda self, node, rng=None: float(len(node)))
    d = bivariates["synergy"]
    bb = backbone_disclosure(d)
    assert set(bb) == {1, 2}
