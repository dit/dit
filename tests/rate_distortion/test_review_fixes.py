"""
Regression tests for rate-distortion bugs found in review.
"""

import numpy as np
import pytest

import dit
from dit.multivariate import entropy
from dit.rate_distortion.blahut_arimoto import blahut_arimoto
from dit.rate_distortion.curves import IBCurve
from dit.rate_distortion.rate_distortion import RateDistortionHamming, RateDistortionMaximumCorrelation


def test_rate_distortion_respects_bound():
    d = dit.Distribution(["0", "1", "2", "3"], [0.4, 0.3, 0.2, 0.1])
    rd = RateDistortionHamming(d, beta=1.0, bound=2)
    assert rd._aux_vars[0].bound == 2


@pytest.mark.parametrize("sizes", [(3, 2), (3, 3)])
def test_conditional_maximum_correlation_axes(sizes):
    from dit.divergences import maximum_correlation

    np.random.seed(0)
    d = dit.random_distribution(2, [range(sizes[0]), range(sizes[1])])
    rd = RateDistortionMaximumCorrelation(d, beta=1.0, rv=[0], crvs=[1])
    x = rd.construct_random_initial()
    full = rd.construct_distribution(x, cutoff=0)
    expected = 1 - maximum_correlation(full, [[0], [2]], [1])
    assert rd.distortion(rd.construct_joint(x)) == pytest.approx(expected)


def test_blahut_arimoto_finds_constant_channel():
    """
    At beta = 0.5 the trivial (zero-rate) point beats the uniform start's fixed point.
    """
    p = np.array([0.4, 0.3, 0.2, 0.1])
    result, _ = blahut_arimoto(p, 0.5)
    assert result.rate + 0.5 * result.distortion <= 0.3 + 1e-9


def test_blahut_arimoto_large_beta_is_finite():
    """
    ``2^{-beta d}`` underflowed to 0/0 once every distortion in a row was large.
    """
    from dit.rate_distortion.distortions import residual_entropy_distortion

    np.random.seed(0)
    p = np.array([0.5, 0.3, 0.2])
    result, _ = blahut_arimoto(p, 1500.0, distortion=residual_entropy_distortion, restarts=3)
    assert np.isfinite(result.rate)
    assert np.isfinite(result.distortion)


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_blahut_arimoto_hamming_matches_restarts(beta):
    """
    Hamming distortion is q-independent, so the two deterministic starts suffice.
    """
    from dit.rate_distortion.blahut_arimoto import _blahut_arimoto
    from dit.rate_distortion.distortions import hamming_distortion

    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4))
    result, _ = blahut_arimoto(p, beta)
    lagrangian = result.rate + beta * result.distortion
    for _ in range(10):
        start = rng.dirichlet(np.ones(4), size=4)
        other, _ = _blahut_arimoto(p, beta, start, hamming_distortion)
        assert lagrangian <= other.rate + beta * other.distortion + 1e-9


def test_ib_curve_entropies_are_of_t():
    np.random.seed(0)
    d = dit.Distribution(["00", "02", "12", "21", "22"], [1 / 5] * 5)
    curve = IBCurve(d, beta_max=5, beta_num=4)
    expected = [entropy(q, [3]) for q in curve.distributions]
    assert np.allclose(curve.entropies, expected, atol=1e-6)


def test_find_kinks_ignores_wraparound():
    curve = IBCurve.__new__(IBCurve)
    curve.betas = np.arange(5.0)
    curve.ranks = np.array([1, 2, 2, 3, 3])
    assert list(curve.find_kinks()) == [2.0]
