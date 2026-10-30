"""
Tests for dit.pid.measures.ideg (degradation intersection information I_d^∩).
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures import ideg
from dit.pid.measures.ideg import PID_Deg
from dit.pid.measures.immi import PID_MMI
from dit.pid.measures.iprec import PID_Prec

RED = ((0,), (1,))


def _random_dist(seed, k):
    """A random distribution with three variables of alphabet size k."""
    rng = np.random.default_rng(seed)
    outcomes = ["".join(map(str, o)) for o in product(range(k), repeat=3)]
    return Distribution(outcomes, rng.dirichlet(np.full(k**3, 0.5)))


def _random_channels(seed, sizes, n_t=3, zeros=False):
    """Random channels from a target of size n_t to sources of the given sizes."""
    rng = np.random.default_rng(seed)
    channels = [rng.dirichlet(np.full(k, 0.6), size=n_t) for k in sizes]
    if zeros:
        for ch in channels:
            ch[rng.random(ch.shape) < 0.3] = 0
            ch[ch.sum(axis=1) == 0, 0] = 1
            ch /= ch.sum(axis=1, keepdims=True)
    return channels


def test_pid_deg_and():
    """
    Test I_d^∩ on the AND distribution.
    Paper Table 2: I_d^∩(AND) = 0.311 bits.
    """
    d = bivariates["and"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.3113, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)


def test_pid_deg_sum():
    """
    Test I_d^∩ on the sum distribution.
    Paper Table 2: I_d^∩(SUM) = 0.5 bits.
    """
    d = bivariates["sum"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)


def test_pid_deg_unique():
    """
    Test I_d^∩ on the unique-1 distribution (T = Y2, Y1 independent).
    Paper Table 2: I_d^∩ = 0.
    """
    d = bivariates["unique 1"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)


def test_pid_deg_redundant():
    """All information is redundant when sources are copies of target."""
    d = bivariates["redundant"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(1.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-3)


def test_pid_deg_synergy():
    """All information is synergistic for XOR."""
    d = bivariates["synergy"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


def test_pid_deg_cat():
    """
    Copy/cat distribution: T = (Y1, Y2) with independent sources.
    I_d^∩ = Gács-Körner common info = 0 (Kolchinsky 2022).
    """
    d = bivariates["cat"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(1.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)


def test_pid_deg_leq_mmi():
    """
    I_d^∩ ≤ I_MMI for all distributions (paper eq. 8-9).
    """
    for name in ["and", "sum", "redundant", "synergy", "unique 1", "diff", "reduced or", "cat"]:
        d = bivariates[name]
        deg = PID_Deg(d, ((0,), (1,)), (2,))
        mmi = PID_MMI(d, ((0,), (1,)), (2,))
        r_deg = deg[((0,), (1,))]
        r_mmi = mmi[((0,), (1,))]
        assert r_deg <= r_mmi + 1e-3, f"I_d^∩ > I_MMI on '{name}': {r_deg:.4f} > {r_mmi:.4f}"


def test_pid_deg_consistent():
    """PID_Deg is consistent on standard distributions."""
    for name in ["and", "sum", "redundant", "synergy", "unique 1"]:
        d = bivariates[name]
        pid = PID_Deg(d, ((0,), (1,)), (2,))
        assert pid.consistent, f"PID_Deg not consistent on '{name}'"


@pytest.mark.parametrize("name", ["and", "diff", "boom", "pnt. unq.", "rdn xor", "gband", "erase", "prob. 2"])
def test_pid_deg_equals_pid_prec(name):
    """I_d^∩ and I_≺ are the same measure."""
    d = bivariates[name]
    assert PID_Deg(d).get_red(RED) == pytest.approx(PID_Prec(d).get_red(RED), abs=1e-9)


def test_pid_deg_beyond_local_search():
    """
    The local search stops short of the maximum on this distribution, while the
    linear program agrees with exhaustive vertex enumeration.
    """
    pytest.importorskip("pypoman")
    d = _random_dist(83, 3)
    exact = PID_Deg(d).get_red(RED)
    local = PID_Deg(d, method="local").get_red(RED)
    vertices, _ = PID_Prec.get_solution_information(d, RED, (2,), n_q=5)
    assert exact == pytest.approx(vertices, abs=1e-8)
    assert exact == pytest.approx(0.0640938, abs=1e-6)
    assert local <= exact + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_pid_deg_local_search_lower_bound(seed):
    """The local search never exceeds the exact value."""
    d = _random_dist(seed, 3)
    exact = PID_Deg._measure(d, RED, (2,))
    assert PID_Deg._measure(d, RED, (2,), method="local") <= exact + 1e-9


def test_pid_deg_method():
    """The local search is used when asked for, or when its parameters are given."""
    d = bivariates["and"]
    assert PID_Deg._measure(d, RED, (2,), niter=5) == pytest.approx(0.3113, abs=1e-3)
    assert PID_Deg._measure(d, RED, (2,), method="local", bound=3) == pytest.approx(0.3113, abs=1e-3)
    with pytest.raises(ValueError, match="method"):
        PID_Deg._measure(d, RED, (2,), method="vertex")


@pytest.mark.parametrize(("seed", "sizes"), [(0, (3, 3)), (1, (2, 4)), (2, (3, 4, 2)), (3, (2, 2, 3))])
def test_degradation_lp_witness(seed, sizes):
    """The linear program returns a common degradation Q attaining its value."""
    channels = _random_channels(seed, sizes)
    pi_t = np.random.default_rng(seed).dirichlet(np.ones(3))
    value, lambdas = ideg._degradation_lp(channels, pi_t)
    k_q = [ch @ lam for ch, lam in zip(channels, lambdas, strict=True)]
    for lam in lambdas:
        assert lam.min() >= -1e-12
        assert np.allclose(lam.sum(axis=1), 1)
    for k in k_q[1:]:
        assert np.allclose(k, k_q[0])
    assert ideg._mi_bits(pi_t, k_q[0]) == pytest.approx(value, abs=1e-9)
    assert k_q[0].shape[1] <= sum(k - 1 for k in sizes) + 1


@pytest.mark.parametrize("seed", range(8))
def test_extreme_rays_exact_matches_cdd(seed):
    """The exact double description method finds the same rays as cddlib."""
    pytest.importorskip("cdd")
    sizes = [(2, 3), (3, 3), (2, 2, 3), (3, 2, 2)][seed % 4]
    channels = _random_channels(seed, sizes, zeros=seed % 2 == 1)
    rows, offsets = ideg._cone_equalities(channels)

    def canonical(rays):
        rays = rays / rays.sum(axis=1, keepdims=True)
        return sorted(map(tuple, np.round(rays, 9)))

    exact = ideg._extreme_rays_exact(channels)
    assert canonical(exact) == canonical(ideg._extreme_rays_cdd(rows, offsets[-1]))


def test_extreme_rays_exact_constant_degradation():
    """
    The exact double description method normalizes the channels, so the constant
    degradation (the all-ones vector) stays in the cone when the rows of a channel
    sum to one only up to rounding.
    """
    channels = _random_channels(0, (2, 2))
    channels[1] = channels[1] * (1 + 2**-52)
    rays = ideg._extreme_rays_exact(channels)
    assert rays.shape == (1, 4)
    assert np.allclose(rays[0], 1)


def test_pid_deg_without_cdd(monkeypatch):
    """Without cddlib, the exact double description method gives the same values."""

    def unavailable(*args, **kwargs):
        raise ImportError

    values = [PID_Deg._measure(_random_dist(seed, 3), RED, (2,)) for seed in range(3)]
    monkeypatch.setattr(ideg, "_extreme_rays_cdd", unavailable)
    for seed, value in enumerate(values):
        assert PID_Deg._measure(_random_dist(seed, 3), RED, (2,)) == pytest.approx(value, abs=1e-9)
