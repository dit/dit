"""
Tests for dit.rate_distortion.curves
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.divergences.pmf import relative_entropy
from dit.exceptions import ditException
from dit.rate_distortion.bottleneck_algorithms import bottleneck_result
from dit.rate_distortion.curves import _MAX_BETA_STEPS, IBCurve, PFCurve, RDCurve
from dit.rate_distortion.distortions import maximum_correlation, residual_entropy
from dit.shannon import entropy


@pytest.mark.flaky(reruns=5)
def test_simple_rd_1():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(["0", "1"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, beta_num=10)
    for r, d in zip(rd.rates, rd.distortions, strict=True):
        assert r == pytest.approx(1 - entropy(d))


@pytest.mark.flaky(reruns=5)
def test_simple_rd_2():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(["0", "1"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, rv=[0], beta_num=10, alpha=0.5, distortion=maximum_correlation)
    assert rd.distortions[0] == pytest.approx(1.0)


@pytest.mark.flaky(reruns=5)
def test_simple_rd_3():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    rd = RDCurve(dist, rv=[0], crvs=[1], beta_num=10, alpha=0.5, distortion=maximum_correlation)
    assert rd.distortions[0] == pytest.approx(1.0)


@pytest.mark.flaky(reruns=5)
def test_simple_rd_4():
    """
    Test against know result, using scipy.
    """
    dist = Distribution(["0", "1"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, beta_num=10, alpha=0.0, distortion=residual_entropy)
    assert rd.distortions[0] == pytest.approx(1.0)


@pytest.mark.flaky(reruns=5)
def test_simple_rd_5():
    """
    Test against know result, using blahut-arimoto.
    """
    dist = Distribution(["0", "1"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, beta_num=10, method="ba")
    for r, d in zip(rd.rates, rd.distortions, strict=True):
        assert r == pytest.approx(1 - entropy(d))


@pytest.mark.flaky(reruns=5)
def test_simple_rd_6():
    """
    Test against know result, using blahut-arimoto.
    """
    dist = Distribution(["0", "1"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, rv=[0], beta_num=10, beta_max=None, method="ba", distortion=residual_entropy)
    assert rd.distortions[0] == pytest.approx(2.0)


@pytest.mark.flaky(reruns=5)
def test_simple_ib_1():
    """
    Test against known values.
    """
    dist = Distribution(["00", "02", "12", "21", "22"], [1 / 5] * 5)
    ib = IBCurve(dist, rvs=[[0], [1]], beta_max=10, beta_num=21)
    assert ib.complexities[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.complexities[5] == pytest.approx(0.8, abs=1e-4)
    assert ib.complexities[20] == pytest.approx(1.5129028136502387, abs=1e-4)
    assert ib.relevances[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.relevances[5] == pytest.approx(0.4, abs=1e-4)
    assert ib.relevances[20] == pytest.approx(0.5701613885745838, abs=1e-4)
    assert 3.0 in ib.find_kinks()


@pytest.mark.flaky(reruns=5)
def test_simple_ib_2():
    """
    Test against known values.
    """
    dist = Distribution(["00", "02", "12", "21", "22"], [1 / 5] * 5)
    ib = IBCurve(dist, beta_max=None, beta_num=21, alpha=0.0)
    assert ib.complexities[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.complexities[12] == pytest.approx(0.97095059445466858, abs=1e-4)
    assert ib.complexities[20] == pytest.approx(1.5219280948873621, abs=1e-4)
    assert ib.relevances[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.relevances[12] == pytest.approx(0.4199730940219748, abs=1e-4)
    assert ib.relevances[20] == pytest.approx(0.5709505944546684, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_simple_ib_3():
    """
    Test against known values.
    """
    dist = Distribution(["00", "02", "12", "21", "22"], [1 / 5] * 5)
    ib = IBCurve(dist, beta_max=None, beta_num=21, alpha=0.5)
    assert ib.complexities[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.complexities[5] == pytest.approx(0.8522009308325029, abs=1e-4)
    assert ib.complexities[20] == pytest.approx(1.5219280948873621, abs=1e-4)
    assert ib.relevances[2] == pytest.approx(0.0, abs=1e-4)
    assert ib.relevances[5] == pytest.approx(0.4080081559717983, abs=1e-4)
    assert ib.relevances[20] == pytest.approx(0.5709505944546684, abs=1e-4)


def test_simple_ib_sequential_dib():
    """
    Test deterministic information bottleneck curve via hard reassignment.
    """
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    ib = IBCurve(dist, variant="dib", method="sequential", beta_min=10, beta_max=10, beta_num=1)

    assert ib.complexities[0] == pytest.approx(1.0)
    assert ib.relevances[0] == pytest.approx(1.0)
    assert ib.alphabets[0] == 2


def test_simple_ib_agglomerative_dib():
    """
    Test deterministic information bottleneck curve via greedy agglomeration.
    """
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    ib = IBCurve(dist, variant="dib", method="agglomerative", beta_min=0, beta_max=0, beta_num=1)

    assert ib.complexities[0] == pytest.approx(0.0)
    assert ib.relevances[0] == pytest.approx(0.0)
    assert ib.alphabets[0] == 1


def test_ib_ba_rejects_unsupported_variants():
    """
    Test that BA does not silently ignore unsupported IB settings.
    """
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    dist2 = Distribution(["000", "011", "100", "111"], [1 / 4] * 4)

    with pytest.raises(ditException):
        IBCurve(dist, alpha=0.0, method="ba")

    with pytest.raises(ditException):
        IBCurve(dist, method="ba", divergence=relative_entropy)

    with pytest.raises(ditException):
        IBCurve(dist2, rvs=[[0], [1]], crvs=[2], method="ba")


def test_ib_hard_methods_reject_soft_variants():
    """
    Test that hard solvers are only used for unconditional DIB.
    """
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])

    with pytest.raises(ditException):
        IBCurve(dist, method="sequential")

    with pytest.raises(ditException):
        IBCurve(dist, variant="gib", alpha=0.5, method="agglomerative")


# ── Option validation, alphabet bounds, and the beta search ──────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"variant": "xib"}, "Variant 'xib' not supported"),
        ({"alpha": 1.5}, r"alpha must be in \[0\.0, 1\.0\]"),
        ({"variant": "gib", "alpha": -0.25}, r"alpha must be in \[0\.0, 1\.0\]"),
        ({"bound": 0}, "bound must be positive"),
        ({"method": "nope"}, "Method 'nope' not supported"),
    ],
)
def test_ib_curve_rejects_invalid_options(kwargs, match):
    """
    Test IBCurve option validation, which happens before any optimization.
    """
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    with pytest.raises(ditException, match=match):
        IBCurve(dist, **kwargs)


def test_ib_curve_bound_is_clamped_to_alphabet():
    """
    Test that a bound larger than |X| gives a bottleneck alphabet of size |X|.
    """
    np.random.seed(0)
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    ib = IBCurve(dist, bound=7, beta_min=10, beta_max=10, beta_num=1)

    assert len(ib.distributions[0].alphabet[3]) == 2
    assert ib.complexities[0] == pytest.approx(1.0, abs=1e-6)
    assert ib.relevances[0] == pytest.approx(1.0, abs=1e-6)


def test_ib_curve_hard_methods_respect_bound():
    """
    Test both hard DIB solvers under bound=2 against a brute-force search over all two-cluster partitions.

    Unbounded, three singleton clusters would win at this beta. The variant name is case-insensitive and fixes
    alpha = 0, which the hard solvers require.
    """
    dist = Distribution(["00", "02", "12", "21", "22"], [1 / 5] * 5)
    p_xy = np.asarray([[1, 0, 1], [0, 0, 1], [0, 1, 1]]) / 5
    beta = 10.0
    best = min(
        (
            bottleneck_result(p_xy, np.eye(2)[list(labels)], beta=beta, alpha=0.0)
            for labels in product(range(2), repeat=3)
        ),
        key=lambda result: result.objective,
    )
    singletons = bottleneck_result(p_xy, np.eye(3), beta=beta, alpha=0.0)
    assert singletons.objective < best.objective

    for method in ("agglomerative", "sequential"):
        ib = IBCurve(dist, variant="DIB", alpha=0.7, method=method, bound=2, beta_min=beta, beta_max=beta, beta_num=1)
        assert ib.label == "DIB"
        assert ib.alphabets[0] == 2
        assert ib.complexities[0] == pytest.approx(best.complexity)
        assert ib.relevances[0] == pytest.approx(best.relevance)


def test_ib_curve_beta_search_gives_up_at_step_cap():
    """
    Test that find_max_beta stops after its step cap when the relevance cannot grow.

    A single bottleneck state carries no information at any beta, so neither stopping rule ever fires.
    """
    np.random.seed(0)
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    ib = IBCurve(dist, bound=1, beta_max=None, beta_num=2)

    assert ib.betas[-1] == pytest.approx(1.5**_MAX_BETA_STEPS)
    np.testing.assert_allclose(ib.complexities, 0.0, atol=1e-9)
    np.testing.assert_allclose(ib.relevances, 0.0, atol=1e-9)
    np.testing.assert_array_equal(ib.alphabets, [1, 1])


def test_rd_curve_beta_search_grows_until_rate_saturates():
    """
    Test that beta_max=None grows beta geometrically until the rate reaches H[X], along R(D) = 1 - h(D).
    """
    np.random.seed(0)
    dist = Distribution(["0", "1"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, beta_num=5, beta_max=None)

    steps = np.log(rd.betas[-1]) / np.log(1.5)
    assert steps == pytest.approx(round(steps))
    assert round(steps) >= 2
    assert rd.rates[-1] == pytest.approx(1.0, abs=1e-4)
    for r, d in zip(rd.rates, rd.distortions, strict=True):
        assert r == pytest.approx(1 - entropy(d), abs=1e-5)


def test_rd_curve_side_information_determining_the_source():
    """
    Test a conditional curve whose side information determines the source: it is the single point (0, 0).

    Its rate never becomes positive, and the beta search must still terminate within its step cap.
    """
    np.random.seed(0)
    dist = Distribution(["00", "11"], [1 / 2, 1 / 2])
    rd = RDCurve(dist, rv=[0], crvs=[1], beta_num=3, beta_max=None)

    np.testing.assert_allclose(rd.rates, 0.0, atol=1e-9)
    np.testing.assert_allclose(rd.distortions, 0.0, atol=1e-9)
    assert 1.5 <= rd.betas[-1] <= 1.5**_MAX_BETA_STEPS * (1 + 1e-9)


def test_pf_curve_conditional():
    """
    Test a conditional privacy funnel curve, which has no perfect-privacy point to time-share with.

    Given Z, S = X xor Z is a relabeling of X, so every release leaks exactly its utility.
    """
    np.random.seed(0)
    dist = Distribution([f"{int(x) ^ int(z)}{x}{z}" for x in "01" for z in "01"], [1 / 4] * 4)
    curve = PFCurve(dist, crvs=[2], rate_num=3)

    assert curve.perfect_utility is None
    np.testing.assert_allclose(curve.rates, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(curve.utilities, curve.rates, atol=1e-5)
    np.testing.assert_allclose(curve.leakages, curve.utilities, atol=1e-5)
