"""
Tests for dit.multivariate.common_informations.tension_common_information
"""

import pytest
from hypothesis import HealthCheck, given, settings

import dit
from dit.multivariate import (
    entanglement,
    entropy,
    gk_common_information,
    tension_common_information,
    total_correlation,
)
from dit.rate_distortion import ShapeFunction
from dit.utils.testing import distributions

giant_bit = dit.Distribution(["00", "11"], [0.5, 0.5])
dsbs = dit.Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
blocks = dit.Distribution(["00", "01", "10", "11", "22"], [0.2] * 5)
independent = dit.Distribution(["00", "01", "10", "11"], [0.25] * 4)


def fano():
    """Uniform on the point-line incidences of the Fano plane."""
    lines = ["012", "034", "056", "136", "145", "235", "246"]
    return dit.Distribution([p + str(i) for i, line in enumerate(lines) for p in line], [1 / 21] * 21)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    ("dist", "value"),
    [
        (giant_bit, 0.0),
        (blocks, 0.0),
        (independent, 0.0),
        (dit.example_dists.Xor(), 1.0),
    ],
)
def test_entanglement_known_values(dist, value):
    """Test against known values."""
    assert float(entanglement(dist, niter=2, maxiter=400)) == pytest.approx(value, abs=1e-4)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [giant_bit, dsbs, blocks, independent])
def test_entanglement_bounds(dist):
    """0 <= E <= min{H[X|Y], H[Y|X], I} and E <= I - K."""
    e = float(entanglement(dist, niter=2, maxiter=400))

    h_joint = float(entropy(dist))
    mutual = float(total_correlation(dist))
    ceiling = min(mutual, *(h_joint - float(entropy(dist, [i])) for i in range(2)))

    assert e >= -1e-6
    assert e <= ceiling + 1e-4
    assert e <= mutual - float(gk_common_information(dist)) + 1e-4


@pytest.mark.flaky(reruns=3)
def test_entanglement_two_ways():
    """E = 2L - 3 S(1/3, 1/3), computed via the shape function and directly."""
    d = dsbs
    h_x, h_y = (float(entropy(d, [i])) for i in range(2))
    ell = float(entropy(d)) - (h_x + h_y) / 2

    shape = ShapeFunction(d, alphas=[[1 / 3, 1 / 3]], niter=2, maxiter=500)
    via_shape = 2 * ell - 3 * shape.values[0]
    direct = float(entanglement(d, niter=2, maxiter=500))

    assert via_shape == pytest.approx(direct, abs=1e-3)


@pytest.mark.flaky(reruns=3)
def test_fano_plane_is_maximally_non_extractable():
    """The Fano incidence structure has E = I, the largest possible."""
    d = fano()
    mutual = float(total_correlation(d))

    assert mutual == pytest.approx(1.2224, abs=1e-4)
    assert float(entanglement(d, niter=2, maxiter=500)) == pytest.approx(mutual, abs=1e-3)
    assert float(tension_common_information(d, niter=2, maxiter=500)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [giant_bit, dsbs, blocks, dit.example_dists.Xor()])
def test_theta_sandwich(dist):
    """(n - 1) K <= Theta <= T."""
    theta = float(tension_common_information(dist, niter=2, maxiter=400))
    n = dist.outcome_length()

    assert theta >= (n - 1) * float(gk_common_information(dist)) - 1e-4
    assert theta <= float(total_correlation(dist)) + 1e-6


@pytest.mark.flaky(reruns=3)
def test_theta_is_continuous_where_gk_is_not():
    """Perturbing the block structure destroys K but leaves Theta positive."""
    perturbed = dit.Distribution(["00", "01", "10", "11", "22", "02"], [0.2, 0.2, 0.2, 0.2, 0.19, 0.01])
    perturbed.normalize()

    assert float(gk_common_information(perturbed)) == pytest.approx(0.0, abs=1e-6)
    assert float(tension_common_information(perturbed, niter=2, maxiter=400)) > 0.1


def test_entanglement_needs_two_sources():
    """A single source has no tension."""
    with pytest.raises(dit.exceptions.ditException, match="at least two sources"):
        entanglement(dsbs, rvs=[[0]])


@pytest.mark.flaky(reruns=3)
def test_conditional_entanglement():
    """Conditioning on a copy of the pair removes all tension."""
    d = dit.Distribution(["000", "111"], [0.5, 0.5])
    assert float(entanglement(d, rvs=[[0], [1]], crvs=[2], niter=2, maxiter=400)) == pytest.approx(0.0, abs=1e-4)


@settings(deadline=None, max_examples=5, suppress_health_check=[HealthCheck.too_slow])
@given(dist=distributions(alphabets=((2, 3),) * 2, nondegenerate=True))
def test_theta_sandwich_property(dist):
    """The sandwich holds on arbitrary pairs."""
    theta = float(tension_common_information(dist, niter=1, maxiter=250))

    assert theta >= float(gk_common_information(dist)) - 1e-3
    assert theta <= float(total_correlation(dist)) + 1e-6
