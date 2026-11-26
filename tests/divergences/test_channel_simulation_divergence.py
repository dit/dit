"""
Tests for dit.divergences.channel_simulation_divergence.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.divergences import channel_simulation_divergence, kullback_leibler_divergence
from dit.divergences.channel_simulation_divergence import channel_simulation_divergence_pmf
from dit.divergences.pmf import channel_simulation_divergence as channel_simulation_divergence_alias
from dit.divergences.pmf import relative_entropy


def _random_pairs(seed, count):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = rng.integers(2, 7)
        p = rng.dirichlet(np.full(n, 0.5))
        p[p < 0.03] = 0
        yield p / p.sum(), rng.dirichlet(np.ones(n))


def test_alias():
    assert channel_simulation_divergence_alias is channel_simulation_divergence_pmf


@pytest.mark.parametrize("alpha", [0.3, 0.6, 1])
def test_self_divergence(alpha):
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert channel_simulation_divergence_pmf(p, p, alpha) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.3, 0.6, 1])
def test_conditioning_on_an_event(alpha):
    """When dP/dQ is constant on its support, w is uniform and every order gives -log2 Q(A) = D(P || Q)."""
    q = np.array([0.1, 0.2, 0.3, 0.4])
    for event in ([0], [1, 3], [0, 2, 3]):
        p = np.zeros(4)
        p[event] = q[event] / q[event].sum()
        assert channel_simulation_divergence_pmf(p, q, alpha) == pytest.approx(-np.log2(q[event].sum()))


def test_hand_computed():
    """r = (0.4, 1.6): w = 1 on (0, 0.4] and 1/2 on (0.4, 1.6], so h(w) = 1.2 * eta(1/2) = 0.6."""
    assert channel_simulation_divergence_pmf([0.2, 0.8], [0.5, 0.5]) == pytest.approx(0.6)
    expected = 2 * np.log2(0.4 + 1.2 * 0.5**0.5)
    assert channel_simulation_divergence_pmf([0.2, 0.8], [0.5, 0.5], 0.5) == pytest.approx(expected)


def test_kullback_leibler_sandwich():
    """D <= D_CS <= D + log2(D + 1) + 1 (Proposition 1 of Flamich et al.)."""
    for p, q in _random_pairs(0, 80):
        kl, dcs = relative_entropy(p, q), channel_simulation_divergence_pmf(p, q)
        assert kl - 1e-12 <= dcs <= kl + np.log2(kl + 1) + 1


def test_order_monotone():
    """Differential Renyi entropies decrease with the order."""
    for p, q in _random_pairs(1, 40):
        values = [channel_simulation_divergence_pmf(p, q, alpha) for alpha in (0.2, 0.5, 0.8, 1)]
        assert np.all(np.diff(values) <= 1e-12)


def test_convex_in_first_argument():
    """P -> D_CS(P || Q) is convex (Proposition 2 of Flamich et al.)."""
    rng = np.random.default_rng(2)
    for _ in range(40):
        p1, p2, q = rng.dirichlet(np.ones(5), size=3)
        lam = rng.uniform()
        mixed = channel_simulation_divergence_pmf(lam * p1 + (1 - lam) * p2, q)
        bound = lam * channel_simulation_divergence_pmf(p1, q) + (1 - lam) * channel_simulation_divergence_pmf(p2, q)
        assert mixed <= bound + 1e-12


def test_not_absolutely_continuous():
    assert channel_simulation_divergence_pmf([0.5, 0.5], [1.0, 0.0]) == np.inf
    assert channel_simulation_divergence_pmf([0.5, 0.5], [1.0, 0.0], 0.5) == np.inf


def test_distributions():
    """P uniform on half of Q's (uniform) outcomes: one bit, as for the KL divergence."""
    d1 = Distribution(["00", "01"], [0.5, 0.5])
    d2 = Distribution(["00", "01", "10", "11"], [0.25] * 4)
    assert channel_simulation_divergence(d1, d2) == pytest.approx(1)
    assert channel_simulation_divergence(d1, d2) == pytest.approx(kullback_leibler_divergence(d1, d2))
    assert channel_simulation_divergence(d2, d1) == np.inf
    assert channel_simulation_divergence(d1, d2, rvs=[1]) == pytest.approx(0)


def test_units():
    d1 = Distribution(["0", "1"], [0.2, 0.8])
    d2 = Distribution(["0", "1"], [0.5, 0.5])
    d1.set_base("e")
    assert channel_simulation_divergence(d1, d2) == pytest.approx(0.6 * np.log(2))


@pytest.mark.parametrize("alpha", [0, 2, -1])
def test_alpha_validation(alpha):
    with pytest.raises(ValueError, match="order"):
        channel_simulation_divergence_pmf([0.5, 0.5], [0.5, 0.5], alpha)
