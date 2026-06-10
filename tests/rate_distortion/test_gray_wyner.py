"""
Tests for dit.rate_distortion.gray_wyner
"""

import numpy as np
import pytest

import dit
from dit.multivariate import (
    entropy,
    exact_common_information,
    gk_common_information,
    kamath_common_information,
    wyner_common_information,
)
from dit.rate_distortion import (
    GrayWynerCurve,
    GrayWynerNetwork,
    lossy_wyner_common_information,
)
from dit.rate_distortion.gray_wyner import GrayWynerOptimizer, hamming_matrix


def giant_bit():
    """A perfectly correlated pair, X == Y."""
    return dit.Distribution(["00", "11"], [0.5, 0.5])


def dsbs():
    """A doubly-symmetric binary source."""
    return dit.Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])


def test_hamming_matrix():
    """The Hamming matrix is 1 - I."""
    np.testing.assert_array_equal(hamming_matrix(3), 1 - np.eye(3))


def test_lazy_plotter_attribute():
    """`GrayWynerPlotter` is lazily importable from the package."""
    import dit.rate_distortion.gray_wyner as gw
    from dit.rate_distortion.gray_wyner.plotting import GrayWynerPlotter

    assert gw.GrayWynerPlotter is GrayWynerPlotter


def test_unknown_attribute_raises():
    """Accessing an unknown attribute raises AttributeError."""
    import dit.rate_distortion.gray_wyner as gw

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = gw.NotARealAttribute


def test_corner_points_giant_bit():
    """All common informations of the giant bit equal H(X) = 1."""
    corners = GrayWynerNetwork(giant_bit()).corner_points(niter=3, maxiter=300)
    for name, value in corners.items():
        assert float(value) == pytest.approx(1.0, abs=1e-4), name


def test_corner_points_delegate():
    """The network's corners match the canonical common informations."""
    d = dsbs()
    gw = GrayWynerNetwork(d)
    corners = gw.corner_points(niter=4, maxiter=400)

    assert float(corners["gacs_korner"]) == pytest.approx(float(gk_common_information(d)), abs=1e-4)
    assert float(corners["kamath"]) == pytest.approx(float(kamath_common_information(d)), abs=1e-4)
    assert float(corners["wyner"]) == pytest.approx(float(wyner_common_information(d, niter=4, maxiter=400)), abs=1e-2)
    assert float(corners["exact"]) == pytest.approx(float(exact_common_information(d, niter=4, maxiter=400)), abs=1e-2)


def test_rate_point_endpoints():
    """Pure-common weight gives R_0 = 0; equal weights collapse onto a vertex."""
    d = dsbs()
    gw = GrayWynerNetwork(d)

    # weighting only the common rate drives R_0 -> 0
    pt = gw.rate_point([1, 0, 0], niter=3, maxiter=300)
    assert pt.common == pytest.approx(0.0, abs=1e-4)

    # weighting only a private rate drives that private rate -> 0
    pt = gw.rate_point([0, 1, 0], niter=3, maxiter=300)
    assert pt.private[0] == pytest.approx(0.0, abs=1e-4)


def test_region_validity():
    """Every region point is non-negative and on/above the Pangloss face."""
    d = dsbs()
    gw = GrayWynerNetwork(d)
    h_joint = float(entropy(d))
    h_sum = float(entropy(d, [0])) + float(entropy(d, [1]))

    points = gw.region(num=8, niter=2, maxiter=250, seed=1)

    sums = []
    for p in points:
        assert p.common >= -1e-4
        assert all(r >= -1e-4 for r in p.private)
        total = p.common + sum(p.private)
        # the sum rate can never beat the joint entropy, nor exceed the
        # independent encoding of every source.
        assert total >= h_joint - 1e-3
        assert total <= h_sum + 1e-3
        sums.append(total)

    # at least one swept point lands on the minimum sum-rate (Pangloss) face.
    assert min(sums) == pytest.approx(h_joint, abs=1e-2)


def test_lossless_budget_matches_none():
    """A zero distortion budget is identical to the lossless network."""
    d = dsbs()
    dm = [hamming_matrix(2), hamming_matrix(2)]

    lossless = GrayWynerNetwork(d).rate_point([1, 1, 1], niter=3, maxiter=300)
    zero_budget = GrayWynerNetwork(d, distortions=dm, bounds=[0.0, 0.0]).rate_point([1, 1, 1], niter=3, maxiter=300)

    # equal weights minimize the total rate (the well-defined quantity); the
    # individual R_0 split along the min sum-rate face is not unique.
    lossless_total = lossless.common + sum(lossless.private)
    zero_total = zero_budget.common + sum(zero_budget.private)
    assert zero_total == pytest.approx(lossless_total, abs=1e-2)


def test_lossy_wyner_reduces_to_wyner():
    """At D = 0 the lossy common information is the Wyner common information."""
    d = dsbs()
    lossy = float(lossy_wyner_common_information(d, niter=4, maxiter=400))
    canonical = float(wyner_common_information(d, niter=4, maxiter=400))
    assert lossy == pytest.approx(canonical, abs=1e-2)


def test_lossy_wyner_decreasing():
    """The lossy common information decreases as distortion is allowed."""
    d = dsbs()
    dm = [hamming_matrix(2), hamming_matrix(2)]

    c0 = float(lossy_wyner_common_information(d, niter=4, maxiter=400))
    c_big = float(lossy_wyner_common_information(d, bounds=[0.45, 0.45], distortions=dm, niter=5, maxiter=500))
    assert c_big < c0 - 0.1


def test_nvariate_ordering():
    """For a 3-source distribution, K <= Wyner <= Exact."""
    d = dit.example_dists.Xor()
    corners = GrayWynerNetwork(d).corner_points(niter=4, maxiter=400)
    assert float(corners["gacs_korner"]) <= float(corners["wyner"]) + 1e-4
    assert float(corners["wyner"]) <= float(corners["exact"]) + 1e-4


def test_curve():
    """The trade-off curve runs and has sensible endpoints."""
    d = dsbs()
    c = GrayWynerCurve(d, s_min=0.0, s_max=3.0, s_num=5, niter=2, maxiter=250)
    h_joint = float(entropy(d))
    h_sum = float(entropy(d, [0])) + float(entropy(d, [1]))

    assert c.r0s.shape == (5,)
    # low private weight: nothing common, everything private
    assert c.r0s[0] == pytest.approx(0.0, abs=1e-3)
    assert c.private_totals[0] == pytest.approx(h_sum, abs=1e-2)
    # high private weight: everything common
    assert c.r0s[-1] == pytest.approx(h_joint, abs=1e-2)
    assert c.private_totals[-1] == pytest.approx(0.0, abs=1e-2)


def test_lambdas_validation():
    """Bad weight vectors are rejected."""
    d = dsbs()
    with pytest.raises(dit.exceptions.ditException):
        GrayWynerOptimizer(d, [1, 1])  # wrong length (n + 1 = 3)
    with pytest.raises(dit.exceptions.ditException):
        GrayWynerOptimizer(d, [1, -1, 1])  # negative weight
