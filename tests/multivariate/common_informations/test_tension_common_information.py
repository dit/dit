"""
Tests for dit.multivariate.common_informations.tension_common_information
"""

import pytest
from hypothesis import HealthCheck, given, settings

import dit
from dit.multivariate import (
    dual_total_correlation,
    entanglement,
    entropy,
    gk_common_information,
    tension_common_information,
    total_correlation,
)
from dit.multivariate.common_informations.tension_common_information import EntanglementOptimizer
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
        (dit.example_dists.Xor(), 2.0),
        (dit.Distribution(["000", "111"], [0.5, 0.5]), 0.0),
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
@pytest.mark.parametrize(
    "dist",
    [
        giant_bit,
        dsbs,
        blocks,
        dit.example_dists.Xor(),
        dit.Distribution(["000", "111", "001", "110"], [0.4, 0.4, 0.1, 0.1]),
    ],
)
def test_theta_sandwich(dist):
    """K <= Theta <= B."""
    theta = float(tension_common_information(dist, niter=2, maxiter=400))

    assert theta >= float(gk_common_information(dist)) - 1e-4
    assert theta <= float(dual_total_correlation(dist)) + 1e-6


@pytest.mark.flaky(reruns=3)
def test_theta_giant_bit_is_on_the_common_information_scale():
    """On the three-variable giant bit Theta = K = B = 1, not the total correlation 2."""
    d = dit.Distribution(["000", "111"], [0.5, 0.5])
    assert float(total_correlation(d)) == pytest.approx(2.0)
    assert float(tension_common_information(d, niter=2, maxiter=400)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.flaky(reruns=3)
def test_xor_is_maximally_non_extractable():
    """Every probe leaves the whole dual total correlation of XOR as tension."""
    d = dit.example_dists.Xor()
    assert float(tension_common_information(d, niter=2, maxiter=400)) == pytest.approx(0.0, abs=1e-4)


def noisy_copies(q, n):
    """n noisy copies of a latent fair bit through i.i.d. BSC(q)."""
    from itertools import product

    pmf = {}
    for z in (0, 1):
        for bits in product((0, 1), repeat=n):
            w = 0.5
            for b in bits:
                w *= q if b != z else 1 - q
            key = "".join(map(str, bits))
            pmf[key] = pmf.get(key, 0.0) + w
    return dit.Distribution(list(pmf), list(pmf.values()))


def stack(*dists):
    """Independent product, variable i of the result being the tuple of the factors' variable i."""
    n = dists[0].outcome_length()
    d = dists[0]
    for d2 in dists[1:]:
        d = d.__matmul__(d2)
    return d.coalesce([[i + n * j for j in range(len(dists))] for i in range(n)])


@pytest.mark.flaky(reruns=3)
def test_meet_split_on_products():
    """Revealing the giant bit is free, so Theta(giant x noisy) = 1 + Theta(noisy)."""
    noisy = noisy_copies(0.1, 3)
    giant = dit.Distribution(["000", "111"], [0.5, 0.5])
    theta = float(tension_common_information(noisy, niter=2, maxiter=500))

    assert theta == pytest.approx(0.0024466, abs=1e-5)
    assert float(tension_common_information(stack(giant, noisy), niter=2, maxiter=500)) == pytest.approx(
        1 + theta, abs=1e-5
    )


@pytest.mark.flaky(reruns=3)
def test_private_randomness_is_irrelevant():
    """Appending an independent private bit to every source changes neither E nor Theta."""
    noisy = noisy_copies(0.1, 3)
    private = dit.Distribution(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)

    assert float(entanglement(stack(noisy, private), niter=2, maxiter=500)) == pytest.approx(
        float(entanglement(noisy, niter=2, maxiter=500)), abs=1e-5
    )


@pytest.mark.flaky(reruns=3)
def test_direct_optimizer_matches_gray_wyner_for_pairs():
    """At n = 2 the direct probe optimizer and the Gray-Wyner query agree."""
    opt = EntanglementOptimizer(dsbs, rvs=[[0], [1]], bound=6)
    opt.optimize(niter=2, maxiter=500)
    direct = float(opt.objective(opt._optima))

    assert direct == pytest.approx(float(entanglement(dsbs, niter=2, maxiter=500)), abs=1e-3)


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
    assert theta <= float(dual_total_correlation(dist)) + 1e-6


@settings(deadline=None, max_examples=3, suppress_health_check=[HealthCheck.too_slow])
@given(dist=distributions(alphabets=(2,) * 3))
def test_theta_sandwich_property_three_sources(dist):
    """K <= Theta <= B on arbitrary binary triples."""
    theta = float(tension_common_information(dist, niter=1, maxiter=250))

    assert theta >= float(gk_common_information(dist)) - 1e-3
    assert theta <= float(dual_total_correlation(dist)) + 1e-6
