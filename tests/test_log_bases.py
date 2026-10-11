"""
Measures on a log-base distribution are reported in that base's units.

Log-valued measures computed on ``d.copy(base=b)`` equal the linear result
divided by ``log2(b)``; unitless measures are unchanged.
"""

import numpy as np
import pytest

import dit
from dit.distconst import mixture_distribution, product_distribution
from dit.divergences import (
    alpha_divergence,
    bhattacharyya_coefficient,
    chernoff_information,
    copy_mutual_information,
    cross_entropy,
    earth_movers_distance,
    hellinger_distance,
    hellinger_divergence,
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    renyi_divergence,
    tsallis_divergence,
    variational_distance,
)
from dit.multivariate import (
    coinformation,
    dual_total_correlation,
    entropy,
    functional_common_information,
    mss_common_information,
    total_correlation,
    transmission,
    wyner_common_information,
)
from dit.other import (
    LMPR_complexity,
    cumulative_residual_entropy,
    lautum_information,
    maximal_leakage,
    renyi_entropy,
    sibson_mutual_information,
    tsallis_entropy,
)
from dit.pid import PID_BROJA, PID_MMI, PID_WB
from dit.profiles import EntropyTriangle

P = dit.Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
Q = dit.Distribution(["00", "01", "10", "11"], [0.25] * 4)
AND = dit.Distribution(["000", "010", "100", "111"], [0.3, 0.2, 0.15, 0.35])
NUMERIC = dit.Distribution([1, 2, 3, 5], [0.1, 0.2, 0.3, 0.4])

LOG_PAIRWISE = [
    cross_entropy,
    kullback_leibler_divergence,
    lambda p, q: jensen_shannon_divergence([p, q]),
    chernoff_information,
    lambda p, q: renyi_divergence(p, q, 2),
]

UNITLESS_PAIRWISE = [
    variational_distance,
    hellinger_distance,
    bhattacharyya_coefficient,
    earth_movers_distance,
    lambda p, q: tsallis_divergence(p, q, 2),
    lambda p, q: alpha_divergence(p, q, 0.5),
    lambda p, q: hellinger_divergence(p, q, 1),
]

LOG_SINGLE = [
    entropy,
    total_correlation,
    dual_total_correlation,
    coinformation,
    transmission,
    lambda d: renyi_entropy(d, 2),
    lautum_information,
    functional_common_information,
    mss_common_information,
    lambda d: (np.random.seed(0), wyner_common_information(d, niter=1))[1],
    lambda d: copy_mutual_information(d, [0], [1]),
    lambda d: sibson_mutual_information(d, [0], [1], 2),
    lambda d: maximal_leakage(d, [0], [1]),
    lambda d: PID_WB(d)[((0, 1),)],
    lambda d: PID_MMI(d)[((0, 1),)],
    lambda d: PID_BROJA(d)[((0,),)],
]

UNITLESS_SINGLE = [
    lambda d: tsallis_entropy(d, 2),
    LMPR_complexity,
    lambda d: EntropyTriangle(d).points[0],
]

BASES = [2, "e", 10]


def _scale(base):
    return 1 / np.log2(P.copy(base=base).get_base(numerical=True))


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("measure", LOG_PAIRWISE)
def test_log_pairwise_follow_base(measure, base):
    expected = measure(P, Q) * _scale(base)
    assert np.allclose(measure(P.copy(base=base), Q.copy(base=base)), expected)


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("measure", UNITLESS_PAIRWISE)
def test_unitless_pairwise_base_invariant(measure, base):
    assert np.allclose(measure(P.copy(base=base), Q.copy(base=base)), measure(P, Q))


@pytest.mark.parametrize("base", [2, "e"])
@pytest.mark.parametrize("measure", LOG_SINGLE)
def test_log_single_follow_base(measure, base):
    expected = measure(AND) * _scale(base)
    assert np.allclose(measure(AND.copy(base=base)), expected, atol=1e-4)


@pytest.mark.parametrize("base", [2, "e"])
@pytest.mark.parametrize("measure", UNITLESS_SINGLE)
def test_unitless_single_base_invariant(measure, base):
    assert np.allclose(measure(AND.copy(base=base)), measure(AND))


def test_cre_follows_base():
    expected = cumulative_residual_entropy(NUMERIC) / np.log2(np.e)
    assert np.isclose(cumulative_residual_entropy(NUMERIC.copy(base="e")), expected)


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


def test_units_attached_in_base(monkeypatch):
    pytest.importorskip("pint")
    from dit.utils.units import ureg

    monkeypatch.setitem(dit.ditParams, "units", True)
    h = entropy(Q.copy(base="e"))
    assert h.units == ureg.nat
    assert np.isclose(h.to(ureg.bit).magnitude, 2.0)
    nested = coinformation(AND)
    assert nested.units == ureg.bit
