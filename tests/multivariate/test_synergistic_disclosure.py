"""
Tests for the standalone synergistic disclosure scalar functions.
"""

import pytest

from dit.distconst import uniform
from dit.multivariate.synergistic_disclosure import (
    backbone_disclosure,
    self_synergy,
    synergistic_disclosure,
)
from dit.pid.distributions import bivariates


# ─────────────────────────────────────────────────────────────────────────────
# synergistic_disclosure
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergistic_disclosure_xor():
    d = bivariates["synergy"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergistic_disclosure_copy():
    d = bivariates["redundant"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[[0], [1]])
    assert val == pytest.approx(0.0, abs=1e-4)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergistic_disclosure_empty_alpha():
    d = bivariates["synergy"]
    val = synergistic_disclosure(d, [[0], [1]], [2], alpha=[])
    assert val == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# backbone_disclosure
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_backbone_disclosure_xor():
    d = bivariates["synergy"]
    bb = backbone_disclosure(d)
    assert bb[2] == pytest.approx(1.0, abs=1e-3)
    assert bb[1] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_backbone_disclosure_copy():
    d = bivariates["redundant"]
    bb = backbone_disclosure(d)
    assert bb[1] == pytest.approx(1.0, abs=1e-3)
    assert bb[2] == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# self_synergy
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_self_synergy_two_coins():
    d = uniform(["00", "01", "10", "11"])
    val = self_synergy(d, sources=[[0], [1]])
    assert val == pytest.approx(1.0, abs=1e-3)
