"""
Tests for the exact computation of the alpha-synergies of synergistic disclosure.
"""

import numpy as np
import pytest

import dit.pid.syndisc as syndisc
from dit import Distribution
from dit.pid.distributions import bivariates, trivariates
from dit.pid.syndisc import ModifiedSynDisc, SynDisc, _alpha_synergy_exact, _choose_method

AND_SYNERGY = 1.5 - 0.75 * np.log2(3)  # 0.3113 in Table 1 of Rosas et al. (2020)


def _random(shape, seed=1):
    pmf = np.random.default_rng(seed).dirichlet(np.ones(int(np.prod(shape))))
    return Distribution([tuple(str(i) for i in o) for o in np.ndindex(*shape)], pmf)


def test_exact_and():
    """The synergy of AND, in closed form."""
    sd = SynDisc(bivariates["and"], method="exact")
    assert sd.get_synergy(((0,), (1,))) == pytest.approx(AND_SYNERGY, abs=1e-12)
    assert ModifiedSynDisc(bivariates["and"], method="exact")[((0,), (1,))] == pytest.approx(AND_SYNERGY, abs=1e-12)


def test_exact_anddup():
    """
    In anddup the third source copies the first, so constraining the third as well as the
    first changes nothing: these nodes have the synergy of AND.
    """
    sd = SynDisc(trivariates["anddup"], method="exact")
    for node in [((0,), (1,)), ((0,), (2,)), ((1,), (2,)), ((0,), (1,), (2,))]:
        assert sd.get_synergy(node) == pytest.approx(AND_SYNERGY, abs=1e-12)


def test_exact_xor_and_copy():
    xor = SynDisc(bivariates["synergy"], method="exact")
    assert xor.get_synergy(((0,), (1,))) == pytest.approx(1.0, abs=1e-12)
    copy = SynDisc(bivariates["redundant"], method="exact")
    assert copy.get_synergy(((0,), (1,))) == pytest.approx(0.0, abs=1e-12)


def test_exact_monotonicity_and_sums():
    """S_alpha decreases down the lattice and the atoms add up to I(X : Y)."""
    d = _random((2, 3, 2, 3))
    sd = SynDisc(d, method="exact")
    for node in sd._lattice:
        for desc in sd._lattice.descendants(node):
            assert sd.get_synergy(desc) <= sd.get_synergy(node) + 1e-12
    assert sum(sd.get_atom(node) for node in sd._lattice) == pytest.approx(sd._total, abs=1e-9)


def test_exact_without_cddlib(monkeypatch):
    """The exact double description gives the same values as cddlib."""
    d = _random((3, 2, 2, 3))
    with_cdd = SynDisc(d, method="exact")
    monkeypatch.setattr(syndisc, "_verified_cdd_rays", lambda rows, dim, tol=1e-9: None)
    without_cdd = SynDisc(d, method="exact")
    for node in with_cdd._lattice:
        assert without_cdd.get_synergy(node) == pytest.approx(with_cdd.get_synergy(node), abs=1e-9)


def test_exact_grouped_sources():
    """Sources made of several variables are coalesced."""
    d = _random((2, 2, 2, 3))
    grouped = _alpha_synergy_exact(d, [[0, 1], [2]], [3], ((0,), (1,)))
    coalesced = d.coalesce([[0, 1], [2], [3]])
    assert grouped == pytest.approx(_alpha_synergy_exact(coalesced, [[0], [1]], [2], ((0,), (1,))), abs=1e-12)


@pytest.mark.flaky(reruns=5)
def test_exact_bounds_optimizer():
    """
    The exact value is the maximum: the optimizer's value can exceed it only by the slack
    it allows in the constraints.
    """
    d = bivariates["and"]
    exact = SynDisc(d, method="exact")
    optimized = SynDisc(d, method="optimize")
    for node in exact._lattice:
        assert optimized.get_synergy(node) <= exact.get_synergy(node) + 2e-3


def test_choose_method(monkeypatch):
    small = bivariates["and"]
    large = _random((7, 5, 2))  # 35 joint values of the sources
    sources = ((0,), (1,))
    assert _choose_method(small, sources, "auto") == "exact"
    assert _choose_method(large, sources, "auto") == "optimize"
    assert _choose_method(large, sources, "exact") == "exact"
    assert _choose_method(small, sources, "optimize") == "optimize"
    monkeypatch.setattr(syndisc, "_exact_limit", lambda: 3)
    assert _choose_method(small, sources, "auto") == "optimize"
    with pytest.raises(ValueError, match="method"):
        _choose_method(small, sources, "simplex")
