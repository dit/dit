"""
Tests for dit.rate_distortion.gray_wyner.extension.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import dit
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
    ep = ExtensionProfile(dsbs(0.2), seed=0)
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
    h_k, _ = korner_graph_entropy(ExtensionProfile(dsbs(0.2)).pxy)
    assert h_k == pytest.approx(1, abs=1e-6)
    # A common bit plus independent private bits: two cliques, so H_K = H[X] - 1.
    d = dit.Distribution([(g, a, g, b) for g in "01" for a in "01" for b in "01"], [1 / 8] * 8)
    h_k, _ = korner_graph_entropy(ExtensionProfile(d, rvs=[[0, 1], [2, 3]]).pxy)
    assert h_k == pytest.approx(1, abs=1e-6)


def test_named_points_dsbs():
    c = 0.3
    ep = ExtensionProfile(dsbs(c), seed=1)
    # Gács-Körner: no common part, so W = const.
    assert np.allclose(ep.gacs_korner, ep.probes["W = const"])
    # Kamath: the MSS of Y for X is Y itself.
    _, values = ep.kamath
    assert values["G Y -> X"] == pytest.approx(1)
    assert values["necessary conditional entropy Y -> X"] == pytest.approx(_h(c))
    # DSBS(0.3) is maximally non-extractable: E = I.
    _, e = ep.entanglement
    assert e == pytest.approx(ep.mutual_information, abs=1e-5)
    # The interaction informations all reach min{H[X|Y], H[Y|X]} via W = X xor Y.
    point, _ = ep.interaction_information
    assert point[2] == pytest.approx(1, abs=1e-5)


def test_wyner_point_dsbs():
    c = 0.3
    ep = ExtensionProfile(dsbs(c), seed=2)
    p = ep.wyner(niter=2)
    q = (1 - np.sqrt(1 - 2 * c)) / 2
    assert p[2] == pytest.approx(0, abs=1e-6)
    assert ep.h_xy - p[0] - p[1] == pytest.approx(1 + _h(c) - 2 * _h(q), abs=1e-3)


def test_lower_boundary_reaches_wyner_rim():
    c = 0.1
    ep = ExtensionProfile(dsbs(c), seed=3)
    low = ep.lower_boundary(num_theta=8, num_radii=9).reshape(-1, 3)  # 8 angles include theta = pi/4
    assert all(within_ext_max(ep, p) for p in low)
    q = (1 - np.sqrt(1 - 2 * c)) / 2
    bottom = low[low[:, 2] == 0]
    assert bottom[:, :2].sum(1).max() == pytest.approx(2 * _h(q), abs=1e-3)


def test_information_bottleneck_is_mrs_gerber():
    """On a DSBS the IB arc is the Mrs. Gerber curve x = h(c * h^{-1}(y))."""
    c = 0.1
    ep = ExtensionProfile(dsbs(c), seed=4)
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
    ep = ExtensionProfile(dsbs(0.2), seed=5)
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
        "probe W's",
    ):
        assert name in names


def test_plot_degenerate_body():
    """X = Y: the profile is the segment {(t, t, t)}, so there is no surface to draw."""
    pytest.importorskip("plotly")
    ep = ExtensionProfile(giant_bit(), seed=6)
    names = {t.name for t in ep.plot(_quick(ep)).data}
    assert "lower boundary of Ext" not in names
    assert "Gács–Körner" in names
