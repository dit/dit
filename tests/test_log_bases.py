"""
Measures must give the same answer for a distribution and its log-base copy.
"""

import numpy as np
import pytest

import dit
from dit.distconst import mixture_distribution, product_distribution
from dit.divergences import (
    alpha_divergence,
    bhattacharyya_coefficient,
    chernoff_information,
    cross_entropy,
    earth_movers_distance,
    hellinger_distance,
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    renyi_divergence,
    tsallis_divergence,
    variational_distance,
)
from dit.multivariate import functional_common_information, mss_common_information
from dit.other import lautum_information, renyi_entropy, tsallis_entropy
from dit.pid import PID_WB
from dit.profiles import EntropyTriangle

P = dit.Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
Q = dit.Distribution(["00", "01", "10", "11"], [0.25] * 4)
AND = dit.Distribution(["000", "010", "100", "111"], [0.3, 0.2, 0.15, 0.35])

PAIRWISE = [
    cross_entropy,
    kullback_leibler_divergence,
    lambda p, q: jensen_shannon_divergence([p, q]),
    variational_distance,
    hellinger_distance,
    bhattacharyya_coefficient,
    chernoff_information,
    earth_movers_distance,
    lambda p, q: renyi_divergence(p, q, 2),
    lambda p, q: tsallis_divergence(p, q, 2),
    lambda p, q: alpha_divergence(p, q, 0.5),
]

SINGLE = [
    lambda d: renyi_entropy(d, 2),
    lambda d: tsallis_entropy(d, 2),
    lautum_information,
    functional_common_information,
    mss_common_information,
    lambda d: PID_WB(d)[((0, 1),)],
    lambda d: EntropyTriangle(d).points[0],
]


@pytest.mark.parametrize("base", [2, "e", 10])
@pytest.mark.parametrize("measure", PAIRWISE)
def test_pairwise_base_invariant(measure, base):
    assert np.allclose(measure(P, Q), measure(P.copy(base=base), Q.copy(base=base)))


@pytest.mark.parametrize("base", [2, "e"])
@pytest.mark.parametrize("measure", SINGLE)
def test_single_base_invariant(measure, base):
    assert np.allclose(measure(AND), measure(AND.copy(base=base)))


def test_log_constructor_fills_gaps_with_log_zero():
    d = dit.Distribution(["00", "11"], [-1, -1], base=2, sparse=False, trim=False)
    d.validate()
    assert np.allclose(d.copy(base="linear").pmf, [0.5, 0, 0, 0.5])


def test_log_mixture():
    a = dit.Distribution(["00", "11"], [0.5, 0.5])
    b = dit.Distribution(["01", "10"], [0.5, 0.5])
    mix = mixture_distribution([a.copy(base=2), b.copy(base=2)], [-1, -1])
    assert np.allclose(mix.copy(base="linear").pmf, [0.25] * 4)


def test_product_distribution_keeps_base():
    d = dit.example_dists.Xor().copy(base=2)
    pd = product_distribution(d)
    assert pd.get_base() == 2
    assert np.allclose(pd.copy(base="linear").pmf, [1 / 8] * 8)
