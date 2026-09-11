"""
Tests for MarginalLiftProfile and marginal-lift fitting.
"""

import pytest

from dit import Distribution
from dit.algorithms import fit_marginal_lift_mixture, marginal_lift_dists
from dit.divergences import kullback_leibler_divergence as D
from dit.profiles import MarginalLiftProfile


def test_copy_puts_mass_on_pair():
    copy = Distribution(["000", "001", "110", "111"], [0.25] * 4)
    fit = fit_marginal_lift_mixture(copy, order=2, mode="uniform")
    assert D(copy, fit["dist"]) == pytest.approx(0.0, abs=1e-6)
    # label (0, 1) should carry essentially all weight
    idx = fit["labels"].index((0, 1))
    assert fit["alpha"][idx] == pytest.approx(1.0, abs=1e-3)


def test_xor_needs_full_joint():
    xor = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    fit2 = fit_marginal_lift_mixture(xor, order=2)
    fit3 = fit_marginal_lift_mixture(xor, order=3)
    assert fit2["L2"] > 0.2
    assert fit3["L2"] == pytest.approx(0.0, abs=1e-6)
    assert D(xor, fit3["dist"]) == pytest.approx(0.0, abs=1e-6)


def test_marginal_lift_profile_copy():
    copy = Distribution(["000", "001", "110", "111"], [0.25] * 4)
    prof = MarginalLiftProfile(copy, mode="uniform")
    assert set(prof.profile.keys()) == {1, 2, 3}
    # Residual should be ~0 by order 2
    assert prof.residuals[2] == pytest.approx(0.0, abs=1e-5)
    assert all(v >= -1e-6 for v in prof.profile.values())


def test_giant_bit_order3():
    gb = Distribution(["000", "111"], [0.5, 0.5])
    dists, metas = marginal_lift_dists(gb, k_max=3)
    assert metas[2]["L2"] > 0.3
    assert metas[3]["L2"] == pytest.approx(0.0, abs=1e-6)


def test_product_lift_mode():
    w = Distribution(["001", "010", "100"], [1 / 3] * 3)
    prof = MarginalLiftProfile(w, mode="product", n_init=8)
    assert prof.residuals[-1] == pytest.approx(0.0, abs=1e-5)


def test_unknown_lift_mode_and_negative_order():
    from dit.algorithms.marginal_lifts import lift_marginal
    from dit.exceptions import ditException

    d = Distribution(["000", "111"], [0.5, 0.5])
    with pytest.raises(ditException):
        fit_marginal_lift_mixture(d, order=-1)
    with pytest.raises(ditException):
        # Build cartesian table then hit lift_marginal with bad mode.
        from dit.algorithms.marginal_lifts import _cartesian

        outs, pmf, alph, _ = _cartesian(d)
        lift_marginal(outs, pmf, alph, (0,), mode="nope")


def test_marginal_lift_named_rvs_and_order0():
    d = Distribution(["000", "111"], [0.5, 0.5])
    d.set_rv_names("XYZ")
    dists, metas = marginal_lift_dists(d, k_max=2, mode="uniform", n_init=4)
    assert dists[0].get_rv_names() == ("X", "Y", "Z")
    assert metas[0]["L2"] is None
    fit = fit_marginal_lift_mixture(d, order=3)
    assert fit["L2"] == pytest.approx(0.0, abs=1e-6)
