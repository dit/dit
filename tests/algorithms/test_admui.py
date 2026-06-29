"""
Tests for dit.algorithms.admui.
"""

import numpy as np
import pytest

import dit
from dit.algorithms.admui import admui_dist, admui_optimize
from dit.algorithms.broja_method import broja_solve_bivariate
from dit.pid.distributions import bivariates


def _scipy_uniques(d):
    uniques, _ = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="scipy")
    return uniques


@pytest.mark.parametrize("name", ["and", "diff", "redundant"])
def test_admui_matches_scipy_bivariates(name):
    d = bivariates[name]
    ref = _scipy_uniques(d)
    got, _meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="admui")
    for key in ref:
        assert got[key] == pytest.approx(ref[key], abs=1e-4)


def test_admui_and_distribution():
    d = dit.Distribution(["000", "001", "010", "111"], [0.25] * 4)
    q, meta = admui_dist(d, [[0], [1]], [2])
    assert meta["converged"]
    assert np.isclose(q.pmf.sum(), 1.0)


def test_admui_xor_reference():
    px = np.array([[2.0 / 3, 0.0], [1.0 / 3, 1.0]])
    py = px.copy()
    ps = np.array([0.75, 0.25])
    q, n_iters, converged = admui_optimize(px, py, ps)
    assert converged
    assert n_iters < 1000
    assert np.isclose(q.sum(), 1.0)


COMPUTE_UI_CASES = {
    "xor": (["000", "011", "101", "110"], [0.25] * 4),
    "and": (["000", "001", "010", "111"], [0.25] * 4),
    "perturbed_xor": (["000", "011", "101", "110"], [0.1, 0.2, 0.3, 0.4]),
}


@pytest.mark.parametrize("case", ["xor", "and"])
def test_admui_computeui_matches_scipy(case):
    outcomes, pmf = COMPUTE_UI_CASES[case]
    d = dit.Distribution(outcomes, pmf)
    ref = _scipy_uniques(d)
    got, _ = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="admui")
    for key in ref:
        assert got[key] == pytest.approx(ref[key], abs=1e-4)


def test_admui_computeui_perturbed_xor():
    """perturbed_xor has multiple BROJA local optima; scipy branch is platform-dependent."""
    outcomes, pmf = COMPUTE_UI_CASES["perturbed_xor"]
    d = dit.Distribution(outcomes, pmf)
    got, meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="admui")
    assert meta["converged"]
    # admUI consistently selects the lower-u0 branch (scipy can return u0 ~ 0.03).
    assert got[(0,)] < 1e-4
    assert got[(1,)] == pytest.approx(0.08966, rel=1e-3)
