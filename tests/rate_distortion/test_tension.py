"""
Tests for the tension region, extension profile, and shape function.
"""

import numpy as np
import pytest

import dit
from dit.multivariate import (
    entropy,
    gk_common_information,
    total_correlation,
    wyner_common_information,
)
from dit.rate_distortion import GrayWynerNetwork, ShapeFunction
from dit.rate_distortion.gray_wyner import (
    GrayWynerOptimizer,
    GrayWynerPoint,
    asymmetric_private_interaction_information,
    extension_to_rates,
    maximal_interaction_information,
    mutual_information_to_rates,
    rates_to_extension,
    rates_to_mutual_information,
    rates_to_tension,
    symmetric_private_interaction_information,
    tension_to_rates,
)


def giant_bit():
    """A perfectly correlated pair; its support graph is a perfect matching."""
    return dit.Distribution(["00", "11"], [0.5, 0.5])


def dsbs():
    """A doubly-symmetric binary source."""
    return dit.Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])


def blocks():
    """Two components: a 2x2 block and an isolated edge, so K = log(2.5)."""
    return dit.Distribution(["00", "01", "10", "11", "22"], [0.2] * 5)


def cycle(n=3):
    """The uniform distribution on a 2n-cycle."""
    outcomes = [f"{i}{j}" for i in range(n) for j in (i, (i + 1) % n)]
    return dit.Distribution(outcomes, [1 / (2 * n)] * (2 * n))


def path():
    """A three-edge path: contains a path of length three, but no cycle."""
    return dit.Distribution(["00", "01", "10"], [1 / 3] * 3)


def fano():
    """Uniform on the point-line incidences of the Fano plane."""
    lines = ["012", "034", "056", "136", "145", "235", "246"]
    outcomes = [p + str(i) for i, line in enumerate(lines) for p in line]
    return dit.Distribution(outcomes, [1 / 21] * 21)


def _profile(dist):
    """The marginal entropies and joint entropy of a distribution."""
    marginals = [float(entropy(dist, [i])) for i in range(dist.outcome_length())]
    return marginals, float(entropy(dist))


# ── coordinate transforms ────────────────────────────────────────────────


@pytest.mark.parametrize("dist", [dsbs(), dit.example_dists.Xor()])
def test_tension_round_trip(dist):
    """Gray-Wyner -> tension -> Gray-Wyner is the identity."""
    marginals, joint = _profile(dist)
    rng = np.random.default_rng(0)

    for _ in range(10):
        point = GrayWynerPoint(common=rng.random(), private=tuple(rng.random(len(marginals))))
        back = tension_to_rates(rates_to_tension(point, marginals, joint), marginals, joint)
        assert back.common == pytest.approx(point.common)
        assert back.private == pytest.approx(point.private)


@pytest.mark.parametrize(
    ("forward", "backward", "extra"),
    [
        (rates_to_extension, extension_to_rates, "joint"),
        (rates_to_mutual_information, mutual_information_to_rates, "marginals"),
    ],
)
def test_other_round_trips(forward, backward, extra):
    """The extension-profile and mutual-information maps also invert."""
    marginals, joint = _profile(dit.example_dists.Xor())
    argument = joint if extra == "joint" else marginals
    rng = np.random.default_rng(1)

    for _ in range(10):
        point = GrayWynerPoint(common=rng.random(), private=tuple(rng.random(len(marginals))))
        back = backward(forward(point, argument), argument)
        assert back.common == pytest.approx(point.common)
        assert back.private == pytest.approx(point.private)


def test_trivial_probe_coordinates():
    """The trivial probe W = . sits at zero tension with residual T."""
    d = dsbs()
    marginals, joint = _profile(d)

    point = GrayWynerPoint(common=0.0, private=tuple(marginals))
    tension = rates_to_tension(point, marginals, joint)

    assert tension.tensions == pytest.approx((0.0, 0.0))
    assert tension.residual == pytest.approx(float(total_correlation(d)))


def test_single_source_rejected():
    """The tension region needs at least two sources."""
    marginals, joint = [1.0], 1.0
    tension = rates_to_tension(GrayWynerPoint(common=0.0, private=(1.0,)), marginals, joint)
    with pytest.raises(dit.exceptions.ditException, match="at least two sources"):
        tension_to_rates(tension, marginals, joint)


def test_mismatched_lengths():
    """Coordinate tuples must match the number of sources."""
    with pytest.raises(dit.exceptions.ditException, match="but there are"):
        rates_to_tension(GrayWynerPoint(common=0.0, private=(1.0, 1.0)), [1.0], 1.0)


# ── the optimizer's new knobs ────────────────────────────────────────────


def test_signed_weights_gate():
    """Negative weights require opting in."""
    d = dsbs()
    with pytest.raises(dit.exceptions.ditException, match="non-negative"):
        GrayWynerOptimizer(d, [1, -1, 1])
    GrayWynerOptimizer(d, [1, -1, 1], allow_signed=True)


def test_fenchel_eggleston_bound():
    """The cardinality bound is |X||Y| + 2."""
    assert GrayWynerOptimizer(dsbs(), [1, 1, 1]).compute_bound() == 2 * 2 + 2


def test_rate_equality_length_validated():
    """Equality constraints must weight every rate."""
    with pytest.raises(dit.exceptions.ditException, match="must have length 3"):
        GrayWynerOptimizer(dsbs(), [1, 1, 1], rate_equalities=[([1, 1], 1.0)])


# ── the region views ─────────────────────────────────────────────────────


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [giant_bit(), dsbs(), blocks()])
def test_residual_intercept_is_i_minus_k(dist):
    """The residual intercept of the tension region is I - K."""
    network = GrayWynerNetwork(dist)
    intercepts = network.tension_intercepts(niter=2, maxiter=300)

    expected = float(total_correlation(dist)) - float(gk_common_information(dist))
    assert intercepts["residual"] == pytest.approx(expected, abs=1e-3)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [giant_bit(), dsbs(), blocks()])
def test_wyner_from_the_zero_residual_face(dist):
    """C_Wyner = I + min{tau_1 + tau_2 : tau_res = 0}."""
    network = GrayWynerNetwork(dist)
    point = network.rate_point(
        [2.0, 1.0, 1.0],
        niter=2,
        maxiter=400,
        rate_equalities=[([1.0, 1.0, 1.0], network._joint_entropy)],
    )
    tension = rates_to_tension(point, network._marginal_entropies, network._joint_entropy)

    via_face = float(total_correlation(dist)) + sum(tension.tensions)
    canonical = float(wyner_common_information(dist, niter=3, maxiter=400))
    assert via_face == pytest.approx(canonical, abs=2e-3)


@pytest.mark.flaky(reruns=3)
def test_region_views_agree_with_rates():
    """Every view is the same region, read in different coordinates."""
    network = GrayWynerNetwork(dsbs())
    marginals, joint = network._marginal_entropies, network._joint_entropy

    # One sweep, converted four ways: each `region` call re-runs a stochastic
    # optimizer, so comparing separate sweeps would compare different points.
    rates = network.region(num=4, niter=1, maxiter=200, seed=3)

    for point in rates:
        tension = rates_to_tension(point, marginals, joint)
        assert tension.residual >= -1e-4
        assert all(t >= -1e-4 for t in tension.tensions)

        extension = rates_to_extension(point, joint)
        info = rates_to_mutual_information(point, marginals)
        # H[X_i|W] + I[X_i:W] = H[X_i]
        recovered = np.asarray(extension.conditional_entropies) + np.asarray(info.marginals)
        assert recovered == pytest.approx(marginals, abs=1e-9)
        assert extension.residual == pytest.approx(tension.residual)
        assert info.joint == pytest.approx(point.common)


@pytest.mark.flaky(reruns=3)
def test_region_view_methods_run():
    """The network exposes each view directly."""
    network = GrayWynerNetwork(dsbs())
    kwargs = {"num": 3, "niter": 1, "maxiter": 200, "seed": 3}

    assert len(network.tension_region(**kwargs)) == len(network.region(**kwargs))
    assert all(p.residual >= -1e-4 for p in network.extension_profile(**kwargs))
    assert all(p.joint >= -1e-4 for p in network.mutual_information_region(**kwargs))
    assert network.tension_point([1.0, 1.0, 1.0], niter=1, maxiter=200).residual >= -1e-4


# ── the shape function ───────────────────────────────────────────────────


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [dsbs(), blocks()])
def test_shape_shannon_forced_values(dist):
    """S is pinned by the entropy profile at several directions."""
    h_x, h_y = (float(entropy(dist, [i])) for i in range(2))
    h_joint = float(entropy(dist))

    shape = ShapeFunction(dist, alphas=[[1, 0], [0, 1], [0, 0], [0.5, 0.5]], niter=1, maxiter=300)

    assert shape.values[0] == pytest.approx(h_joint - h_y, abs=1e-4)  # H[X|Y]
    assert shape.values[1] == pytest.approx(h_joint - h_x, abs=1e-4)  # H[Y|X]
    assert shape.values[2] == pytest.approx(0.0, abs=1e-6)
    assert shape.values[3] == pytest.approx(h_joint - (h_x + h_y) / 2, abs=1e-4)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [giant_bit(), dsbs(), blocks(), cycle()])
def test_shape_between_its_envelopes(dist):
    """S_min <= S <= S_max at every sampled direction."""
    shape = ShapeFunction(dist, num=4, niter=1, maxiter=250)
    assert np.all(shape.values >= shape.minima - 1e-4)
    assert np.all(shape.values <= shape.maxima + 1e-4)

    report = shape.rigidity()
    assert -1e-3 <= report["min"] <= report["mean"] <= report["max"] <= 1 + 1e-3


@pytest.mark.flaky(reruns=3)
def test_shape_of_an_extractable_pair_is_maximal():
    """A pair whose mutual information is extractable attains S_max."""
    shape = ShapeFunction(blocks(), num=5, niter=1, maxiter=300)
    assert shape.rigidity()["maximal"]


def test_shape_validation():
    """Bad directions and too few sources are rejected."""
    with pytest.raises(dit.exceptions.ditException, match="unit cube"):
        ShapeFunction(dsbs(), alphas=[[2.0, 0.0]])
    with pytest.raises(dit.exceptions.ditException, match="must have 2 columns"):
        ShapeFunction(dsbs(), alphas=[[0.5, 0.5, 0.5]])


# ── the Li-El Gamal extreme points ───────────────────────────────────────


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("dist", [giant_bit(), dsbs(), path(), cycle()])
def test_interaction_information_chain(dist):
    """0 <= G_PPI <= G_PNI <= G_NNI <= min{H[X|Y], H[Y|X]}."""
    kwargs = {"niter": 1, "maxiter": 250}
    ppi = float(symmetric_private_interaction_information(dist, **kwargs))
    pni = float(asymmetric_private_interaction_information(dist, **kwargs))
    nni = float(maximal_interaction_information(dist, **kwargs))

    h_joint = float(entropy(dist))
    ceiling = min(h_joint - float(entropy(dist, [i])) for i in range(2))

    assert ppi >= -1e-4
    assert ppi <= pni + 1e-3
    assert pni <= nni + 1e-3
    assert nni <= ceiling + 1e-3


@pytest.mark.flaky(reruns=3)
def test_interaction_informations_vanish_on_a_matching():
    """A support graph with no path of length three forces all three to zero."""
    kwargs = {"niter": 1, "maxiter": 250}
    assert float(maximal_interaction_information(giant_bit(), **kwargs)) == pytest.approx(0.0, abs=1e-4)
    assert float(symmetric_private_interaction_information(giant_bit(), **kwargs)) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.flaky(reruns=3)
def test_symmetric_vanishes_on_an_acyclic_support():
    """G_PPI is zero exactly on acyclic supports, where G_NNI need not be."""
    kwargs = {"niter": 1, "maxiter": 250}
    assert float(symmetric_private_interaction_information(path(), **kwargs)) == pytest.approx(0.0, abs=1e-4)
    assert float(maximal_interaction_information(path(), **kwargs)) > 1e-2


@pytest.mark.flaky(reruns=3)
def test_interaction_informations_maximal_on_a_cycle():
    """With H[X] = H[Y] and p(x) = p(y) on the support, all three attain H[Y|X]."""
    d = cycle()
    ceiling = float(entropy(d)) - float(entropy(d, [0]))
    kwargs = {"niter": 1, "maxiter": 300}

    for measure in (
        maximal_interaction_information,
        asymmetric_private_interaction_information,
        symmetric_private_interaction_information,
    ):
        assert float(measure(d, **kwargs)) == pytest.approx(ceiling, abs=1e-2), measure.__name__


def test_interaction_informations_need_a_pair():
    """These quantities are defined for two sources only."""
    with pytest.raises(dit.exceptions.ditException, match="2 sources"):
        maximal_interaction_information(dit.example_dists.Xor())


# ── plotting ─────────────────────────────────────────────────────────────


def test_plot_tension_and_shape():
    """The 3D tension view and the shape surface render."""
    import matplotlib

    matplotlib.use("Agg")
    from dit.rate_distortion.gray_wyner.plotting import GrayWynerPlotter

    network = GrayWynerNetwork(dsbs())
    assert GrayWynerPlotter.plot_tension(network, num=3, niter=1, maxiter=200, seed=0) is not None
    assert ShapeFunction(dsbs(), num=3, niter=1, maxiter=200).plot() is not None
    assert ShapeFunction(dit.example_dists.Xor(), num=3, niter=1, maxiter=200).plot() is not None


def test_plot_tension_needs_a_pair():
    """The tension region is only plottable in three dimensions."""
    import matplotlib

    matplotlib.use("Agg")
    from dit.rate_distortion.gray_wyner.plotting import GrayWynerPlotter

    network = GrayWynerNetwork(dit.example_dists.Xor())
    with pytest.raises(dit.exceptions.ditException, match="2 sources"):
        GrayWynerPlotter.plot_tension(network)
