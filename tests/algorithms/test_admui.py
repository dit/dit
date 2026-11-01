"""
Tests for dit.algorithms.admui.
"""

import numpy as np
import pytest

import dit
from dit.algorithms.admui import _conditionals_from_dist, _iproj_gis, _iproj_is, admui_dist, admui_optimize
from dit.algorithms.broja_method import broja_solve_bivariate
from dit.algorithms.pid_broja import prepare_dist
from dit.multivariate import coinformation
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


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("case", ["and"])
def test_admui_computeui_matches_scipy(case):
    outcomes, pmf = COMPUTE_UI_CASES[case]
    d = dit.Distribution(outcomes, pmf)
    ref = _scipy_uniques(d)
    got, _ = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="admui")
    for key in ref:
        assert got[key] == pytest.approx(ref[key], abs=1e-4)


def test_admui_computeui_xor():
    """XOR has zero unique information; scipy can land on a spurious positive branch."""
    outcomes, pmf = COMPUTE_UI_CASES["xor"]
    d = dit.Distribution(outcomes, pmf)
    got, meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="admui")
    assert meta["converged"]
    assert got[(0,)] == pytest.approx(0.0, abs=1e-6)
    assert got[(1,)] == pytest.approx(0.0, abs=1e-6)


def test_admui_computeui_perturbed_xor():
    """perturbed_xor has multiple BROJA local optima; scipy branch is platform-dependent."""
    outcomes, pmf = COMPUTE_UI_CASES["perturbed_xor"]
    d = dit.Distribution(outcomes, pmf)
    got, meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="admui")
    assert meta["converged"]
    # admUI consistently selects the lower-u0 branch (scipy can return u0 ~ 0.03).
    assert got[(0,)] < 1e-4
    assert got[(1,)] == pytest.approx(0.08966, rel=1e-3)


# ── I-projection methods, iteration limits, degenerate targets ─────────


def _joint(d):
    """The pmf of a dense distribution as an array with one axis per variable."""
    return d.pmf.reshape([len(a) for a in d.alphabet])


def _uniques(q):
    """I(X0 : Y | X1) and I(X1 : Y | X0) of a coalesced (X0, X1, Y) distribution."""
    return (
        coinformation(q, [[0], [2]], [1]),
        coinformation(q, [[1], [2]], [0]),
    )


def test_iprojections_agree():
    """
    IS and GIS both return the I-projection of the reference: the rescaling a_x r_xy b_y
    with the requested marginals. Rows of zero marginal probability are dropped.
    """
    r = np.random.default_rng(0).dirichlet(np.ones(12)).reshape(3, 4)
    px = np.array([0.5, 0.0, 0.5])
    py = np.array([0.1, 0.2, 0.3, 0.4])
    projections = [iproj(px, py, r, eps=1e-13, maxiter=10_000) for iproj in (_iproj_is, _iproj_gis)]
    for b, x_mask, y_mask in projections:
        np.testing.assert_array_equal(x_mask, [True, False, True])
        np.testing.assert_array_equal(y_mask, [True] * 4)
        assert b.shape == (2, 4)
        np.testing.assert_allclose(b.sum(axis=1), [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(b.sum(axis=0), py, atol=1e-12)
        log_ratio = np.log(b / r[x_mask])
        additive = log_ratio[:, :1] + log_ratio[:1, :] - log_ratio[0, 0]
        np.testing.assert_allclose(log_ratio, additive, atol=1e-12)
    np.testing.assert_allclose(projections[0][0], projections[1][0], atol=1e-12)


def test_gis_single_step():
    """One GIS iteration multiplies the reference by sqrt(p_x p_y / (r_x r_y))."""
    r = np.random.default_rng(1).dirichlet(np.ones(6)).reshape(2, 3)
    px = np.array([0.3, 0.7])
    py = np.array([0.5, 0.25, 0.25])
    b, _, _ = _iproj_gis(px, py, r, eps=1e-12, maxiter=1)
    rx, ry = r.sum(axis=1), r.sum(axis=0)
    expected = r * np.sqrt(np.outer(px, py) / np.outer(rx, ry))
    np.testing.assert_allclose(b, expected, rtol=1e-12)
    assert abs(b.sum(axis=1) - px).max() > 1e-6  # one step does not yet match the marginals


@pytest.mark.parametrize(
    ("name", "expected"),
    [("sum", (0.0, 0.0)), ("boom", (1 / 3, 1 / 3)), ("wb 3", (0.0, np.log2(3) - 4 / 3))],
)
def test_admui_gis_matches_is(name, expected):
    """
    The generalized iterative scaling I-projection reaches the same BROJA optimum as the
    default iterative scaling, with the source-target marginals of the distribution.
    """
    d = bivariates[name]
    p = _joint(prepare_dist(d, [[0], [1]], [2]))
    results = {}
    for ip_method in ("IS", "GIS"):
        q, meta = admui_dist(d, [[0], [1]], [2], ip_method=ip_method)
        assert meta["converged"]
        np.testing.assert_allclose(_joint(q).sum(axis=1), p.sum(axis=1), atol=1e-8)
        np.testing.assert_allclose(_joint(q).sum(axis=0), p.sum(axis=0), atol=1e-8)
        results[ip_method] = _uniques(q)
    assert results["GIS"] == pytest.approx(results["IS"], abs=1e-6)
    assert results["GIS"] == pytest.approx(expected, abs=1e-5)


def test_admui_unknown_ip_method():
    with pytest.raises(ValueError, match="Unknown ip_method 'newton'; expected 'IS' or 'GIS'"):
        admui_dist(bivariates["and"], [[0], [1]], [2], ip_method="newton")


def test_admui_maxiter_unconverged():
    """
    Out of iterations, admUI says so. The last iterate has total mass one and exactly the
    (X1, Y) marginal, which iterative scaling matches last; the (X0, Y) one nearly.
    """
    d = bivariates["boom"]
    p = _joint(prepare_dist(d, [[0], [1]], [2]))
    _, meta = admui_dist(d, [[0], [1]], [2])
    assert meta["converged"]
    assert meta["n_iters"] > 2

    q, meta = admui_dist(d, [[0], [1]], [2], maxiter=2)
    assert meta == {"converged": False, "n_iters": 2}
    q = _joint(q)
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(q.sum(axis=0), p.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose(q.sum(axis=1), p.sum(axis=1), atol=1e-3)


def test_conditionals_unused_target_value():
    """
    A target value of probability zero gets all-zero conditionals rather than a division by
    zero; the other columns are P(X0 | Y) and P(X1 | Y).
    """
    arr = np.zeros((2, 2, 3))
    arr[0, 0, 0], arr[1, 1, 0] = 0.3, 0.1
    arr[0, 1, 1], arr[1, 0, 1] = 0.2, 0.4
    d = dit.Distribution.from_ndarray(arr)
    d.make_dense()
    px, py, ps = _conditionals_from_dist(d)
    np.testing.assert_allclose(ps, [0.4, 0.6, 0.0])
    np.testing.assert_allclose(px, [[0.75, 1 / 3, 0.0], [0.25, 2 / 3, 0.0]])
    np.testing.assert_allclose(py, [[0.75, 2 / 3, 0.0], [0.25, 1 / 3, 0.0]])
    np.testing.assert_allclose(px * ps, arr.sum(axis=1))
    np.testing.assert_allclose(py * ps, arr.sum(axis=0))
