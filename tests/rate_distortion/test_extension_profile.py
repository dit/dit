"""
Tests for dit.rate_distortion.gray_wyner.extension.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import dit
from dit.example_dists import erasure_channel
from dit.exceptions import ditException
from dit.rate_distortion import ExtensionProfile, excess_functional_information, korner_graph_entropy


def _h(p):
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def dsbs(c):
    """A doubly-symmetric binary source with crossover c."""
    return dit.Distribution(["00", "01", "10", "11"], [(1 - c) / 2, c / 2, c / 2, (1 - c) / 2])


def giant_bit():
    """X == Y, a fair bit."""
    return dit.Distribution(["00", "11"], [0.5, 0.5])


def within_ext_max(profile, p, tol=1e-6):
    """Lemma 3.2.F of Matveev & Romashchenko: the Shannon outer bound."""
    x, y, z = p
    return (
        -tol <= z <= min(x, y) + tol
        and x <= profile.h_x + tol
        and y <= profile.h_y + tol
        and x - z <= profile.h_x_given_y + tol
        and y - z <= profile.h_y_given_x + tol
    )


def test_needs_a_pair():
    d = dit.Distribution(["000", "111"], [0.5, 0.5])
    with pytest.raises(ditException):
        ExtensionProfile(d)
    with pytest.raises(ditException):
        ExtensionProfile(d, rvs=[[0], [1], [2]])
    assert ExtensionProfile(d, rvs=[[0], [1, 2]]).h_y == pytest.approx(1)


def test_probes_are_vertices_of_ext_max():
    ep = ExtensionProfile(dsbs(0.1))
    verts, _ = ep.ext_max
    for p in ep.probes.values():
        assert np.min(np.linalg.norm(verts - p, axis=1)) < 1e-9


def test_support_point_shannon_forced():
    """S(1/2, 1/2) = (H[X|Y] + H[Y|X]) / 2 for every pair."""
    ep = ExtensionProfile(dsbs(0.2), rng=0)
    p = ep.support_point((0.5, 0.5, -1))
    assert 0.5 * p[0] + 0.5 * p[1] - p[2] == pytest.approx((ep.h_x_given_y + ep.h_y_given_x) / 2, abs=1e-6)
    assert within_ext_max(ep, p)


@pytest.mark.parametrize("c", [0.05, 0.1, 0.3])
def test_excess_functional_information_dsbs(c):
    """For a binary Y, Prop. 1 of Li & El Gamal is tight: inf_{W indep X} H[Y|W] = 1 - 2c."""
    psi, height = excess_functional_information(ExtensionProfile(dsbs(c)).pxy)
    assert height == pytest.approx(1 - 2 * c, abs=1e-8)
    assert psi == pytest.approx(1 - 2 * c - (1 - _h(c)), abs=1e-8)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.floats(0.01, 1.0), min_size=6, max_size=6))
def test_excess_functional_information_bounds(weights):
    """0 <= Psi <= log(I + 1) + 4 (the strong functional representation lemma)."""
    pxy = np.reshape(weights, (2, 3)) / np.sum(weights)
    psi, _ = excess_functional_information(pxy)
    mi = sum(pxy[i, j] * np.log2(pxy[i, j] / pxy[i].sum() / pxy[:, j].sum()) for i in range(2) for j in range(3))
    assert -1e-9 <= psi <= np.log2(mi + 1) + 4


def test_korner_graph_entropy():
    # Full support: the characteristic graph is complete, so H_K = H[X].
    assert korner_graph_entropy(ExtensionProfile(dsbs(0.2)).pxy) == pytest.approx(1, abs=1e-6)
    # A common bit plus independent private bits: two cliques, so H_K = H[X] - 1.
    d = dit.Distribution([(g, a, g, b) for g in "01" for a in "01" for b in "01"], [1 / 8] * 8)
    assert korner_graph_entropy(ExtensionProfile(d, rvs=[[0, 1], [2, 3]]).pxy) == pytest.approx(1, abs=1e-6)


def _layered(*pmfs):
    """p(x, y) of independent layers, X and Y each the tuple of the layer variables."""
    p = np.ones((1, 1))
    for q in pmfs:
        p = np.einsum("ab,cd->acbd", p, q).reshape(p.shape[0] * q.shape[0], p.shape[1] * q.shape[1])
    return p


def _dsbs_pmf(c):
    return np.array([[1 - c, c], [c, 1 - c]]) / 2


@settings(max_examples=40, deadline=None)
@given(st.lists(st.sampled_from([0.0, 0.0, 0.1, 0.3, 0.5, 1.0]), min_size=9, max_size=9))
def test_reductions_match_brute_force(weights):
    """The exact reductions agree with the raw programs (sparse supports exercise them)."""
    pxy = np.reshape(weights, (3, 3))
    if pxy.sum() == 0:
        return
    pxy = pxy / pxy.sum()
    psi, _ = excess_functional_information(pxy)
    raw, _ = excess_functional_information(pxy, reduce=False)
    assert psi == pytest.approx(raw, abs=1e-7)
    assert korner_graph_entropy(pxy) == pytest.approx(korner_graph_entropy(pxy, reduce=False), abs=1e-6)


def test_reductions_on_layered_source():
    """A giant bit times a DSBS: small enough for the raw programs to check the reductions."""
    pxy = _layered(np.diag([0.5, 0.5]), _dsbs_pmf(0.2))
    assert excess_functional_information(pxy)[0] == pytest.approx(excess_functional_information(pxy, reduce=False)[0])
    assert korner_graph_entropy(pxy) == pytest.approx(korner_graph_entropy(pxy, reduce=False), abs=1e-6)


def test_reductions_scale_to_the_witness():
    """giant (x) DSBS(0.1) (x) DSBS(0.3) (x) private bits: 16 x 16, far too big unreduced.

    Psi is strictly subadditive here (0.520 < 0.550, the sum over layers), and the
    characteristic graph is two cliques, so H_K = H[X] - 1 = 3.
    """
    pxy = _layered(np.diag([0.5, 0.5]), _dsbs_pmf(0.1), _dsbs_pmf(0.3), np.full((2, 2), 0.25))
    psi, height = excess_functional_information(pxy)
    assert height == pytest.approx(1 + 1.16980, abs=1e-4)
    assert psi == pytest.approx(0.52009, abs=1e-4)
    assert psi < (0.8 - (1 - _h(0.1))) + (0.4 - (1 - _h(0.3))) - 1e-3
    assert korner_graph_entropy(pxy) == pytest.approx(3, abs=1e-6)


def test_named_points_dsbs():
    c = 0.3
    ep = ExtensionProfile(dsbs(c), rng=1)
    # Gács-Körner: no common part, so W = const.
    assert np.allclose(ep.gacs_korner, ep.probes["W = const"])
    # Kamath: the MSS of Y for X is Y itself.
    _, values = ep.kamath
    assert values["G Y -> X"] == pytest.approx(1)
    assert values["necessary conditional entropy Y -> X"] == pytest.approx(_h(c))
    # DSBS(0.3) is maximally non-extractable: E = I.
    _, e = ep.entanglement
    assert e == pytest.approx(ep.mutual_information, abs=1e-5)
    # The interaction informations all reach min{H[X|Y], H[Y|X]} = h(c) via W = X xor Y, at (1, 1, 1).
    points, values = ep.interaction_information
    for key, value in values.items():
        assert value == pytest.approx(_h(c), abs=1e-4), key
        assert np.allclose(points[key], [1, 1, 1], atol=1e-4), key


def test_perfect_privacy():
    # A DSBS has an invertible p(x|y): nothing about Y can be revealed independently of X.
    (_, _), values = ExtensionProfile(dsbs(0.2)).perfect_privacy
    assert values["X -> Y"] == pytest.approx(0, abs=1e-8)
    # Give Y an extra fair bit independent of everything: exactly that bit is free.
    d = dit.Distribution(
        [(x, y, b) for x in "01" for y in "01" for b in "01"],
        [(0.4 if x == y else 0.1) / 2 for x in "01" for y in "01" for b in "01"],
    )
    ep = ExtensionProfile(d, rvs=[[0], [1, 2]])
    points, values = ep.perfect_privacy
    assert values["X -> Y"] == pytest.approx(1, abs=1e-8)
    assert np.allclose(points[0], [ep.h_x, ep.h_y - 1, ep.mutual_information])


def test_channel_synthesis():
    """C(0) is Wyner's common information; enough common randomness brings C(t) down to I[X:Y]."""
    c = 0.1
    ep = ExtensionProfile(dsbs(c), rng=7)
    low = ep.lower_boundary(num_theta=8, num_radii=9)
    ts, curves = ep.channel_synthesis(low)
    q = (1 - np.sqrt(1 - 2 * c)) / 2
    for values, points in curves:
        assert values[0] == pytest.approx(1 + _h(c) - 2 * _h(q), abs=1e-3)
        assert values[-1] == pytest.approx(ep.mutual_information, abs=1e-6)
        assert np.all(np.diff(values) <= 1e-12)
        assert np.all(points[:, 2] == 0)


def test_channel_synthesis_erasure_curve():
    """
    The profile's C(t) tracks the closed-form erasure-channel curve of Cuff (2013), Section II-F, from above: the
    sampled bottom face is an inner approximation, so the curve can only overestimate the rate.
    """
    p = 0.3
    ep = ExtensionProfile(erasure_channel(p), rng=11)
    ts, curves = ep.channel_synthesis(ep.lower_boundary(num_theta=16, num_radii=9))
    rs = np.linspace(1 - p, min(2 * (1 - p), 1), 2001)
    exact = np.array([min(max(r, _h(p) + r * (1 - _h(min((1 - p) / r, 1 - 1e-12))) - t) for r in rs) for t in ts])
    gap = curves[0][0] - exact
    assert gap.min() >= -1e-6
    assert gap.max() <= 3e-2


def test_public_channel_synthesis():
    """The public curve runs from the Wyner point to U = Y, where the rate is I[X:Y]."""
    c = 0.1
    ep = ExtensionProfile(dsbs(c), rng=12)
    ts, curves = ep.public_channel_synthesis(ep.lower_boundary(num_theta=8, num_radii=9))
    q = (1 - np.sqrt(1 - 2 * c)) / 2
    assert ts[0] == pytest.approx(1 + _h(c) - 2 * _h(q), abs=1e-3)
    for values, points in curves:
        assert values[0] == pytest.approx(1 - _h(q), abs=1e-3)
        assert values[-1] == pytest.approx(ep.mutual_information, abs=1e-6)
        assert np.all(np.diff(values) <= 1e-12)
        assert np.all(points[:, 2] == 0)


def test_wyner_point_dsbs():
    c = 0.3
    ep = ExtensionProfile(dsbs(c), rng=2)
    p = ep.wyner(niter=2)
    q = (1 - np.sqrt(1 - 2 * c)) / 2
    assert p[2] == pytest.approx(0, abs=1e-6)
    assert ep.h_xy - p[0] - p[1] == pytest.approx(1 + _h(c) - 2 * _h(q), abs=1e-3)


def test_lower_boundary_reaches_wyner_rim():
    c = 0.1
    ep = ExtensionProfile(dsbs(c), rng=3)
    low = ep.lower_boundary(num_theta=8, num_radii=9).reshape(-1, 3)  # 8 angles include theta = pi/4
    assert all(within_ext_max(ep, p) for p in low)
    q = (1 - np.sqrt(1 - 2 * c)) / 2
    bottom = low[low[:, 2] == 0]
    assert bottom[:, :2].sum(1).max() == pytest.approx(2 * _h(q), abs=1e-3)


def test_information_bottleneck_is_mrs_gerber():
    """On a DSBS the IB arc is the Mrs. Gerber curve x = h(c * h^{-1}(y))."""
    c = 0.1
    ep = ExtensionProfile(dsbs(c), rng=4)
    arc = ep.information_bottleneck(num=8)[0]

    def mgl(s):
        lo, hi = 1e-15, 0.5
        for _ in range(100):
            q = (lo + hi) / 2
            lo, hi = (q, hi) if _h(q) < s else (lo, q)
        return _h(c * (1 - q) + q * (1 - c))

    for x, y, z in arc:
        if 1e-6 < y < 1 - 1e-6:
            assert x == pytest.approx(mgl(y), abs=1e-5)
        assert x - z == pytest.approx(ep.h_x_given_y, abs=1e-6)


def _quick(ep):
    return ep.compute(lower={"num_theta": 8, "num_radii": 5}, upper={"num": 20}, curves={"num": 5}, slow=False)


def test_plot():
    pytest.importorskip("plotly")
    ep = ExtensionProfile(dsbs(0.2), rng=5)
    fig = ep.plot(_quick(ep))
    names = {t.name for t in fig.data}
    for name in (
        "lower boundary of Ext",
        "upper boundary of Ext",
        "Gács–Körner",
        "Kamath edges",
        "bottom-face rim",
        "information bottleneck (both zero-leakage faces)",
        "privacy funnel (both zero-leakage faces)",
        "excess functional information Ψ",
        "Körner graph entropy",
        "channel synthesis C(t)",
        "public channel synthesis C_pub(t)",
        "probe W's",
    ):
        assert name in names
    cs = next(t for t in fig.data if t.name == "channel synthesis C(t)")
    assert "decoder local randomness H[Y|W]" in cs.text[0]


def test_plot_degenerate_body():
    """X = Y: the profile is the segment {(t, t, t)}, so there is no surface to draw."""
    pytest.importorskip("plotly")
    ep = ExtensionProfile(giant_bit(), rng=6)
    names = {t.name for t in ep.plot(_quick(ep)).data}
    assert "lower boundary of Ext" not in names
    assert "Gács–Körner" in names


def test_plot_mutual_information_region():
    pytest.importorskip("plotly")
    ep = ExtensionProfile(dsbs(0.2), rng=8)
    features = _quick(ep)
    fig = ep.plot_mutual_information_region(features)
    names = {t.name for t in fig.data}
    for name in (
        "image of Ext's lower boundary",
        "Gács–Körner",
        "privacy funnel (both zero-leakage faces)",
        "plane v_X + v_Y − v_XY = I (bottom face)",
        "plane v_Y = v_XY (X − Y − U)",
    ):
        assert name in names
    # The affine map sends W = const to the origin and W = XY to (H[X], H[Y], H[XY]).
    probes = next(t for t in fig.data if t.name == "probe W's")
    xyz = np.c_[probes.x, probes.y, probes.z]
    assert np.allclose(xyz[list(features["probes"]).index("W = const")], [0, 0, 0.008])
    assert np.allclose(xyz[list(features["probes"]).index("W = XY")], [ep.h_x, ep.h_y, ep.h_xy + 0.008])


def test_plot_tension_region():
    pytest.importorskip("plotly")
    c = 0.2
    ep = ExtensionProfile(dsbs(c), rng=9)
    fig = ep.plot_tension_region(_quick(ep))
    names = {t.name for t in fig.data}
    assert "region of tension (lower boundary)" in names
    # The channel-synthesis curve lies on tau_3 = 0, with the common randomness used read off tau_1.
    cs = next(t for t in fig.data if t.name == "channel synthesis C(t)")
    assert np.allclose(np.asarray(cs.z) - 0.008, 0, atol=1e-9)
    used = float(cs.text[0].split("I[Y:W|X] = ")[1].split("<br>")[0])
    assert used == pytest.approx(cs.x[0] - 0.008, abs=1e-3)  # the overlay lift also shifts tau_1
    intercepts = next(t for t in fig.data if t.name == "axis intercepts")
    xyz = np.c_[intercepts.x, intercepts.y, np.asarray(intercepts.z) - 0.008]
    # No common part, so tau_3 = I; the MSS of Y for X is Y, so tau_1 = H[Y|X] = h(c).
    assert np.allclose(xyz, [[0, 0, ep.mutual_information], [_h(c), 0, 0], [0, _h(c), 0]], atol=1e-6)


def test_region_plots_degenerate_body():
    pytest.importorskip("plotly")
    ep = ExtensionProfile(giant_bit(), rng=10)
    features = _quick(ep)
    assert ep.plot_mutual_information_region(features).data
    assert "region of tension (lower boundary)" not in {t.name for t in ep.plot_tension_region(features).data}
