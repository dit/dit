"""
Tests for dit.algorithms.broja_cone.
"""

import pytest

import dit
from dit.algorithms.broja_cone import broja_cone_solve
from dit.algorithms.broja_method import broja_solve_bivariate
from dit.pid.distributions import bivariates


def _scipy_uniques(d):
    uniques, _ = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="scipy")
    return uniques


@pytest.mark.parametrize("name", ["and", "diff"])
def test_cone_matches_scipy(name):
    pytest.importorskip("ecos")
    d = bivariates[name]
    ref = _scipy_uniques(d)
    got, meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="cone")
    assert meta["converged"]
    for key in ref:
        assert got[key] == pytest.approx(ref[key], abs=1e-3)


def test_cone_and_distribution():
    pytest.importorskip("ecos")
    d = dit.Distribution(["000", "001", "010", "111"], [0.25] * 4)
    uniques, meta = broja_solve_bivariate(d, ((0,), (1,)), (2,), method="cone")
    assert meta["primal_infeas"] < 1e-5
    assert sum(uniques.values()) >= 0


def test_cone_gap_with_conditional_entropy():
    """
    The duality gap compares the conditional entropy and the dual objective in the same
    units (nats). With the sources constant and the target a uniform bit, H(T | X, Y) =
    ln 2, and the gap used to be reported as ln 2 (1 - ln 2) = 0.213 (#302).
    """
    pytest.importorskip("ecos")
    d = dit.Distribution(["000", "001"], [0.5, 0.5])
    _, meta = broja_cone_solve(d, [[0], [1]], [2])
    assert meta["converged"]
    assert meta["gap"] < 1e-6


@pytest.mark.parametrize("name", sorted(bivariates))
def test_cone_gap_closes(name):
    """
    At the optimum the duality gap of the cone program vanishes.
    """
    pytest.importorskip("ecos")
    _, meta = broja_cone_solve(bivariates[name], [[0], [1]], [2])
    assert meta["gap"] < 1e-6
