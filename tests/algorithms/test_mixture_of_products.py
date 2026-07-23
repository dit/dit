"""
Tests for mixture-of-products fitting and BindingMixtureProfile.
"""

import pytest

from dit import Distribution
from dit.algorithms import fit_mixture_of_products, mixture_of_products_dists
from dit.divergences import kullback_leibler_divergence as D
from dit.multivariate import dual_total_correlation as B
from dit.profiles import BindingMixtureProfile, SharedRandomnessDecomposition
from dit.shannon import entropy as H


def _giant_bit():
    return Distribution(["000", "111"], [0.5, 0.5])


def _xor():
    return Distribution(["000", "011", "101", "110"], [0.25] * 4)


def _w():
    return Distribution(["001", "010", "100"], [1 / 3] * 3)


def _copy():
    return Distribution(["000", "001", "110", "111"], [0.25] * 4)


def _and():
    outs, p = [], []
    for x in "01":
        for y in "01":
            z = "1" if (x == "1" and y == "1") else "0"
            outs.append(x + y + z)
            p.append(0.25)
    return Distribution(outs, p)


def test_fit_k1_is_product():
    d = _xor()
    fit = fit_mixture_of_products(d, k=1, n_init=4, seed=1)
    # Product of marginals: D(P||Q) = T(P) = 1 for XOR.
    assert D(d, fit["dist"]) == pytest.approx(1.0, abs=1e-5)
    assert B(fit["dist"]) == pytest.approx(0.0, abs=1e-5)


def test_giant_bit_exact_at_k2():
    d = _giant_bit()
    fit = fit_mixture_of_products(d, k=2, n_init=8, seed=0)
    assert D(d, fit["dist"]) == pytest.approx(0.0, abs=1e-6)
    assert B(fit["dist"]) == pytest.approx(1.0, abs=1e-5)
    assert fit["I_xv"] == pytest.approx(1.0, abs=1e-5)


def test_xor_needs_k4():
    d = _xor()
    dists, meta = mixture_of_products_dists(d, k_max=4, n_init=10, seed=0, early_stop=False)
    assert meta[0]["forward_kl"] == pytest.approx(1.0, abs=1e-4)
    assert meta[3]["forward_kl"] == pytest.approx(0.0, abs=1e-5)
    assert B(dists[3]) == pytest.approx(2.0, abs=1e-5)


def test_binding_mixture_profile_sums_to_b():
    """ΔB atoms are nonnegative and sum to B(P)."""
    for factory in (_giant_bit, _xor, _w, _copy, _and):
        d = factory()
        prof = BindingMixtureProfile(d, k_max=8, n_init=10, seed=0)
        vals = [prof.profile[k] for k in sorted(prof.profile)]
        assert all(v >= -1e-6 for v in vals)
        assert sum(vals) == pytest.approx(B(d), abs=1e-3)


def test_binding_mixture_giant_bit_mass_at_k2():
    d = _giant_bit()
    prof = BindingMixtureProfile(d, k_max=4, n_init=8, seed=0)
    assert prof.profile[1] == pytest.approx(0.0, abs=1e-5)
    assert prof.profile[2] == pytest.approx(1.0, abs=1e-4)
    # Early stop once D≈0, so higher k may be absent.
    assert prof.forward_kl[-1] == pytest.approx(0.0, abs=1e-6)


def test_binding_mixture_copy_at_k2():
    d = _copy()
    prof = BindingMixtureProfile(d, k_max=4, n_init=8, seed=0)
    assert sum(prof.profile[k] for k in prof.profile if k <= 2) == pytest.approx(1.0, abs=1e-3)


def test_binding_mixture_xor_late():
    d = _xor()
    prof = BindingMixtureProfile(d, k_max=4, n_init=10, seed=0, early_stop=False)
    # Little binding at k=2; most arrives by k=4.
    assert prof.profile[2] < 0.5
    assert sum(prof.profile.values()) == pytest.approx(2.0, abs=1e-3)


def test_shared_randomness_decomposition_default_b():
    d = _giant_bit()
    srd = SharedRandomnessDecomposition(d)
    assert "B" in next(iter(srd.atoms.values()))
    # Fully independent node has B=0; full joint has B=1.
    nodes = list(srd.atoms.keys())
    full = max(nodes, key=lambda n: max((len(b) for b in n), default=0))
    singles = min(nodes, key=lambda n: max((len(b) for b in n), default=0))
    assert srd.atoms[full]["B"] == pytest.approx(1.0, abs=1e-5)
    assert srd.atoms[singles]["B"] == pytest.approx(0.0, abs=1e-5)


def test_shared_randomness_matches_dd_with_b():
    from dit.profiles import DependencyDecomposition

    d = _w()
    srd = SharedRandomnessDecomposition(d)
    dd = DependencyDecomposition(d, measures={"B": B})
    for node in srd.atoms:
        assert srd.atoms[node]["B"] == pytest.approx(dd.atoms[node]["B"], abs=1e-6)


def test_ladder_entropy_decreases():
    d = _and()
    dists, _ = mixture_of_products_dists(d, k_max=3, n_init=8, seed=1)
    ents = [H(q) for q in dists]
    assert ents[0] >= ents[-1] - 1e-6
