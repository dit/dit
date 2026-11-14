"""
Tests for dit.pid.iig.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings
from scipy.optimize import minimize

from dit import Distribution
from dit.multivariate import coinformation
from dit.pid.distributions import bivariates
from dit.pid.measures.iig import PID_IG, ig_double_synergy, ig_synergy
from dit.utils.testing import distributions


def test_pid_ig1():
    """
    Test iproj on a generic distribution.
    """
    d = bivariates["and"]
    pid = PID_IG(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.08283, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.22845, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.22845, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.27155, abs=1e-4)


def test_pid_proj2():
    """
    Test iproj on another generic distribution.
    """
    d = bivariates["reduced or"]
    pid = PID_IG(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(-0.03122, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.34250, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.34250, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.34622, abs=1e-4)


def test_ig_synergy_requires_two_sources():
    """ig_synergy is bivariate-only and rejects a single source."""
    d = bivariates["and"]
    with pytest.raises(ValueError, match="exactly 2 sources"):
        ig_synergy(d, [[0]], [2])


def test_ig_synergy_fuzz_at_full_support():
    """
    On a distribution of full support, a tiny fuzz changes nothing.
    """
    weights = np.arange(1, 9, dtype=float)
    weights /= weights.sum()
    outcomes = [f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    d = Distribution(outcomes, list(weights))

    assert ig_synergy(d, [[0], [1]], [2]) == pytest.approx(ig_synergy(d, [[0], [1]], [2], fuzz=1e-100), abs=1e-12)


def test_ig_synergy_different_supports():
    """
    p(x0, x1) p(y | x0) is positive at 100 and p(x0, x1) p(y | x1) at 010, where the other
    vanishes, so the geodesic is finite only for lambda in [0, 1]. The minimum is at lambda = 1,
    the first of them restricted to their common support. Adding a constant to every
    probability approaches it from above, slowly.
    """
    d = Distribution(["011", "101", "110"], [3 / 5, 3 / 10, 1 / 10])
    exact = 1 / 5 + 3 / 10 * np.log2(4 / 3) + np.log2(37 / 40)
    assert ig_synergy(d, [[0], [1]], [2]) == pytest.approx(exact, abs=1e-12)

    fuzzed = [ig_synergy(d, [[0], [1]], [2], fuzz=fuzz) for fuzz in (1e-50, 1e-100, 1e-200)]
    assert fuzzed[0] > fuzzed[1] > fuzzed[2] > exact + 1e-4


@settings(max_examples=25, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_ig_synergy_below_fuzzed(dist):
    """
    A positive fuzz never lowers the synergy.
    """
    exact = ig_synergy(dist, [[0], [1]], [2])
    assert exact <= ig_synergy(dist, [[0], [1]], [2], fuzz=1e-100) + 1e-12
    assert exact >= -1e-12


def _double_synergy_by_search(dist):
    """
    The double synergy by a direct search over the weights of the four projections, computed
    from the marginals, taking each factor P^w literally: 0 for w > 0, 1 for w = 0, and
    infinite for w < 0 where P = 0, on the outcomes where some projection is positive.
    """
    p = np.zeros((2, 2, 2, 2))
    for outcome, prob in zip(dist.outcomes, dist.pmf, strict=True):
        p[tuple(int(x) for x in outcome)] = prob
    projections = []
    for i, j in product((0, 1), (2, 3)):
        joint_y = p.sum(axis=5 - j, keepdims=True)
        joint_x = p.sum(axis=1 - i, keepdims=True)
        shared = p.sum(axis=(1 - i, 5 - j), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            projections.append(np.where(shared > 0, joint_y * joint_x / shared, 0.0))
    projections = np.array(projections)
    union = (projections > 0).any(axis=0)
    projections, p = projections[:, union], p[union]

    def divergence(u):
        w = np.array([1 - u.sum(), *u])[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            zeros = np.where(w > 0, 0.0, np.where(w == 0, 1.0, np.inf))
            q = np.prod(np.where(projections > 0, projections**w, zeros), axis=0)
        if not np.all(np.isfinite(q)):
            return np.inf
        q /= q.sum()
        return float(np.sum(p[p > 0] * np.log2(p[p > 0] / q[p > 0])))

    starts = [(0.25, 0.25, 0.25), (0.1, 0.4, 0.3), (0.5, 0.2, 0.1)]
    options = {"xatol": 1e-12, "fatol": 1e-15, "maxiter": 20000, "maxfev": 40000}
    return min(minimize(divergence, np.array(x0), method="Nelder-Mead", options=options).fun for x0 in starts)


FULL = Distribution(["".join(o) for o in product("01", repeat=4)], np.arange(1, 17) / 136)
# Two or three of the projections vanish together at some outcomes, and the minimum lies on the
# boundary of the admissible weights.
SPARSE = Distribution(["1001", "0110", "1111", "0000", "0101", "0100"], np.array([2, 3, 5, 4, 4, 4]) / 22)


def test_ig_double_synergy_requires_two_and_two():
    with pytest.raises(ValueError, match="2 sources and 2 targets"):
        ig_double_synergy(FULL, [[0], [1]], [[2]])


@pytest.mark.parametrize("dist", [FULL, SPARSE])
def test_ig_double_synergy_matches_search(dist):
    """
    The double synergy agrees with a direct search that shares none of its code.
    """
    expected = _double_synergy_by_search(dist)
    assert ig_double_synergy(dist, [[0], [1]], [[2], [3]]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("dist", [FULL, SPARSE, bivariates["and"].coalesce([[0], [1], [2], [2]])])
def test_ig_double_synergy_symmetries_and_bounds(dist):
    """
    The double synergy does not depend on the order of the sources, of the targets, or on which
    pair is which; and each projection is a point of the hull, at a divergence of
    I[X_{1-i} : Y_{1-j} | X_i, Y_j].
    """
    syn = ig_double_synergy(dist, [[0], [1]], [[2], [3]])
    assert ig_double_synergy(dist, [[1], [0]], [[2], [3]]) == pytest.approx(syn, abs=1e-8)
    assert ig_double_synergy(dist, [[0], [1]], [[3], [2]]) == pytest.approx(syn, abs=1e-8)
    assert ig_double_synergy(dist, [[2], [3]], [[0], [1]]) == pytest.approx(syn, abs=1e-8)
    for i, j in product((0, 1), (2, 3)):
        assert syn <= coinformation(dist, [[1 - i], [5 - j]], [i, j]) + 1e-9
