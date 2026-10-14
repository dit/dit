"""
Tests for dit.rate_distortion.privacy_funnel.
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import entropy, total_correlation
from dit.rate_distortion import (
    IBCurve,
    PFCurve,
    PrivacyFunnel,
    PrivacyFunnelRate,
    admits_perfect_privacy,
    greedy_privacy_funnel,
    perfect_privacy_utility,
    privacy_funnel,
    privacy_funnel_rate,
)
from dit.utils.testing import distributions

# S is binary, X ternary, so perfect privacy is possible.
perfect = Distribution(["00", "01", "12", "10"], [0.3, 0.2, 0.3, 0.2])


def _h(q):
    return float(-sum(p * np.log2(p) for p in (q, 1 - q) if p > 0))


def _star(a, b):
    return a * (1 - b) + b * (1 - a)


def _bsc(p, delta):
    """
    X ~ Bernoulli(p) and S the output of a BSC(delta) on X, as outcomes "sx".
    """
    outcomes, pmf = [], []
    for x, px in [("0", 1 - p), ("1", p)]:
        for s in "01":
            outcomes.append(s + x)
            pmf.append(px * ((1 - delta) if s == x else delta))
    return Distribution(outcomes, pmf)


def _p_sx(dist):
    joint = dist.coalesce([[0], [1]])
    joint.make_dense()
    return joint.pmf.reshape([len(a) for a in joint.alphabet])


def _mi(joint):
    joint = np.asarray(joint)
    outer = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / outer[mask])).sum())


def test_independent():
    """
    When S and X are independent, X can be released in full without leakage.
    """
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert admits_perfect_privacy(d)
    assert perfect_privacy_utility(d) == pytest.approx(1.0)
    assert privacy_funnel_rate(d, 1.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("rate", [0.25, 0.75])
def test_identical(rate):
    """
    When S = X, every bit of utility is a bit of leakage.
    """
    d = Distribution(["00", "11"], [1 / 2] * 2)
    assert not admits_perfect_privacy(d)
    assert privacy_funnel_rate(d, rate) == pytest.approx(rate, abs=1e-4)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_mr_gerbers_lemma(alpha):
    """
    The binary symmetric privacy funnel has a closed form (Asoodeh & Calmon
    2020, Lemma 4).
    """
    p, delta = 0.3, 0.1
    z = max(alpha, 2 * p)
    rate = _h(p) - alpha * _h(p / z)
    pf = _h(_star(p, delta)) - alpha * _h(_star(delta, p / z)) - (1 - alpha) * _h(delta)
    assert privacy_funnel_rate(_bsc(p, delta), rate) == pytest.approx(pf, abs=1e-4)


def test_markov_identity():
    """
    Since S - X - T, I[X : T] - I[S : T] = I[X : T | S] at the optimum.
    """
    pf = PrivacyFunnelRate(perfect, rate=1.2)
    pf.optimize()
    q = Distribution.from_ndarray(pf.construct_joint(pf._optima))
    s, x, t = [0], [1], [3]
    lhs = total_correlation(q, [x, t]) - total_correlation(q, [s, t])
    assert lhs == pytest.approx(total_correlation(q, [x, t], s), abs=1e-6)
    assert total_correlation(q, [x, t]) >= 1.2 - 1e-5


def test_perfect_privacy():
    """
    The linear program's mapping leaks nothing and achieves g0, which is the
    largest rate with zero leakage.
    """
    assert admits_perfect_privacy(perfect)
    g0, p_t_given_x = perfect_privacy_utility(perfect, return_mechanism=True)
    p_sx = _p_sx(perfect)
    assert np.allclose(p_t_given_x.sum(axis=1), 1)
    assert _mi(p_sx @ p_t_given_x) == pytest.approx(0.0, abs=1e-9)
    assert _mi(p_sx.sum(axis=0)[:, None] * p_t_given_x) == pytest.approx(g0)
    assert privacy_funnel_rate(perfect, g0 - 1e-3) == pytest.approx(0.0, abs=1e-6)
    assert privacy_funnel_rate(perfect, g0 + 0.05) > 1e-4


def test_no_perfect_privacy_binary():
    """
    A binary X with dependent S admits no perfect privacy.
    """
    d = _bsc(0.3, 0.1)
    assert not admits_perfect_privacy(d)
    assert perfect_privacy_utility(d) == pytest.approx(0.0, abs=1e-9)


def test_perfect_privacy_too_large():
    """
    Enumerating the polytope is guarded.
    """
    with pytest.raises(ditException):
        perfect_privacy_utility(perfect, max_candidates=1)


@pytest.mark.parametrize("rate", [0.5, 0.8, 1.1, 1.3])
def test_greedy(rate):
    """
    The greedy merge is feasible, and the optimizer is never worse.
    """
    result = greedy_privacy_funnel(_p_sx(perfect), rate)
    assert result.utility >= rate - 1e-9
    assert result.active == result.p_t_given_x.shape[1]
    assert privacy_funnel_rate(perfect, rate) <= result.leakage + 1e-6


def test_greedy_merges_to_constant():
    """
    With no utility requirement, the greedy merge ends at a constant release.
    """
    result = greedy_privacy_funnel(_p_sx(perfect), 0.0)
    assert result.active == 1
    assert result.leakage == pytest.approx(0.0)
    assert len(result.merges) == 2


def test_greedy_bad_rate():
    with pytest.raises(ditException):
        greedy_privacy_funnel(_p_sx(perfect), 5.0)


def test_lagrangian_endpoints():
    """
    beta = 0 releases nothing about S; beta > 1 releases X.
    """
    leakage, _ = privacy_funnel(perfect, 0.0)
    assert leakage == pytest.approx(0.0, abs=1e-6)
    leakage, utility = privacy_funnel(perfect, 1.5)
    assert utility == pytest.approx(entropy(perfect, [1]), abs=1e-6)
    assert leakage == pytest.approx(total_correlation(perfect, [[0], [1]]), abs=1e-6)


def test_lagrangian_beats_perfect_privacy():
    """
    The Lagrangian is no worse than the perfect privacy mapping.
    """
    beta = 0.05
    leakage, utility = privacy_funnel(perfect, beta)
    assert leakage - beta * utility <= -beta * perfect_privacy_utility(perfect) + 1e-6


def test_conditional():
    """
    Given Z, S = X, so leakage equals utility.
    """
    d = Distribution(["000", "110", "001", "111"], [1 / 4] * 4)
    assert privacy_funnel_rate(d, 0.5, rvs=[[0], [1]], crvs=[2]) == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rvs": [[0]]},
        {"beta": -1.0},
    ],
)
def test_bad_args(kwargs):
    args = {"dist": perfect, "beta": 0.5, **kwargs}
    with pytest.raises(ditException):
        PrivacyFunnel(**args)


def test_bad_rate():
    with pytest.raises(ditException):
        PrivacyFunnelRate(perfect, rate=5.0)


@settings(deadline=None, max_examples=10)
@given(dist=distributions(alphabets=((2, 3), (2, 3))))
def test_invariants(dist):
    """
    0 <= leakage <= min(utility, I[S : X]) and utility <= H[X].
    """
    leakage, utility = privacy_funnel(dist, 0.5)
    assert -1e-6 <= leakage <= min(utility, total_correlation(dist, [[0], [1]])) + 1e-6
    assert utility <= entropy(dist, [1]) + 1e-6


@pytest.fixture(scope="module")
def pf_curve():
    return PFCurve(perfect, rate_num=9)


def test_curve(pf_curve):
    """
    The curve is nondecreasing and convex, flat at zero up to g0, and ends at
    (H[X], I[S : X]).
    """
    leakages = pf_curve.leakages
    assert np.all(np.diff(leakages) >= -1e-5)
    assert np.all(np.diff(leakages, 2) >= -1e-4)
    assert np.allclose(leakages[pf_curve.rates <= pf_curve.perfect_utility], 0, atol=1e-6)
    assert pf_curve.utilities[-1] == pytest.approx(entropy(perfect, [1]), abs=1e-6)
    assert leakages[-1] == pytest.approx(total_correlation(perfect, [[0], [1]]), abs=1e-6)


def test_curve_time_sharing(pf_curve):
    """
    A point stuck at a poor local minimum is lowered to the convex envelope of
    the others, and good points are untouched.
    """
    bad = pf_curve.leakages.copy()
    bad[-2] = pf_curve._max_leakage
    repaired = pf_curve._time_share(pf_curve.utilities, bad)
    assert repaired[-2] < pf_curve._max_leakage - 0.05
    assert np.allclose(np.delete(repaired, -2), np.delete(pf_curve.leakages, -2))


def test_curve_greedy(pf_curve):
    curve = PFCurve(perfect, rate_num=9, method="greedy")
    assert np.all(curve.utilities >= curve.rates - 1e-9)
    assert np.all(pf_curve.leakages <= curve.leakages + 1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "nope"},
        {"method": "greedy", "crvs": [1]},
    ],
)
def test_curve_bad_args(kwargs):
    with pytest.raises(ditException):
        PFCurve(perfect, **kwargs)


class TestRegion:
    """
    The IB and PF curves bound the same region.
    """

    @pytest.fixture(scope="class")
    def ib_curve(self):
        return IBCurve(perfect, rvs=[[1], [0]], beta_num=11)

    def test_envelopes(self, ib_curve, pf_curve):
        plotting = pytest.importorskip("dit.rate_distortion.plotting")
        grid, upper, lower = plotting._region_envelopes(ib_curve, pf_curve)
        assert np.all(upper >= lower - 1e-5)
        assert upper[0] == pytest.approx(0, abs=1e-9)
        assert lower[0] == pytest.approx(0, abs=1e-9)
        assert upper[-1] == pytest.approx(lower[-1])
        assert grid[-1] == pytest.approx(entropy(perfect, [1]))
        assert lower[-1] == pytest.approx(total_correlation(perfect, [[0], [1]]))
        assert np.allclose(lower[grid <= pf_curve.perfect_utility], 0, atol=1e-6)

    def test_envelopes_single(self, ib_curve, pf_curve):
        plotting = pytest.importorskip("dit.rate_distortion.plotting")
        _, upper, lower = plotting._region_envelopes(ib_curve=ib_curve)
        assert lower is None
        assert upper[-1] == pytest.approx(total_correlation(perfect, [[0], [1]]))
        _, upper, lower = plotting._region_envelopes(pf_curve=pf_curve)
        assert upper is None
        with pytest.raises(ditException):
            plotting._region_envelopes()

    def test_plotter_validation(self, ib_curve, pf_curve):
        plotting = pytest.importorskip("dit.rate_distortion.plotting")
        plotter = plotting.BottleneckRegionPlotter(ib_curve=ib_curve, pf_curve=pf_curve)
        assert plotter.ib_curve is ib_curve
        with pytest.raises(ditException):
            plotting.BottleneckRegionPlotter()
        with pytest.raises(ditException):
            plotting.BottleneckRegionPlotter(ib_curve=IBCurve(perfect, beta_num=3), pf_curve=pf_curve)
        other = Distribution(["00", "01", "12", "10"], [0.25, 0.25, 0.25, 0.25])
        with pytest.raises(ditException):
            plotting.BottleneckRegionPlotter(ib_curve=IBCurve(other, rvs=[[1], [0]], beta_num=3), pf_curve=pf_curve)
        with pytest.raises(ditException):
            plotting.BottleneckRegionPlotter(ib_curve=IBCurve(perfect, rvs=[[1], [0]], beta_num=3, alpha=0.0))
