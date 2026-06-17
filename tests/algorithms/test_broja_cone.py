"""
Tests for dit.algorithms.broja_cone.
"""

import pytest

import dit
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
