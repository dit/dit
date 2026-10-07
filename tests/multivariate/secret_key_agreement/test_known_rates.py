"""
Tests against sources whose two-way secret key agreement rate is proven.
"""

from functools import partial

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit import Distribution
from dit.example_dists.intrinsic import (
    bound_information,
    chitambar_bob_speaks,
    chitambar_two_way,
    deterministic_erasure_source,
    gisin_wolf,
    intrinsic_1,
    intrinsic_2,
    intrinsic_3,
    james_problem,
    reversely_degraded,
)
from dit.multivariate import total_correlation
from dit.multivariate.secret_key_agreement import (
    intrinsic_mutual_information,
    iterated_discarding_skar,
    less_noisy_intrinsic_mutual_information,
    lower_intrinsic_mutual_information,
    necessary_intrinsic_mutual_information,
    one_way_skar,
    upper_intrinsic_mutual_information,
)

crossover = st.floats(min_value=0.0, max_value=0.5)


def _bsc_chain(a, b):
    """
    X uniform, Y = BSC_a(X), Z = BSC_b(Y), so X - Y - Z.
    """
    outcomes = [f"{x}{y}{z}" for x, y, z in np.ndindex(2, 2, 2)]
    pmf = [(a if x != y else 1 - a) * (b if y != z else 1 - b) / 2 for x, y, z in np.ndindex(2, 2, 2)]
    return Distribution(outcomes, pmf)


@settings(max_examples=25, deadline=None)
@given(a=crossover, b=crossover, swap=st.booleans())
def test_markov_chain(a, b, swap):
    """
    If X - Y - Z (or Y - X - Z) the rate is I[X:Y|Z], and the lower bound
    I[X:Y] - I[X:Z] already meets it.
    """
    d = _bsc_chain(a, b)
    rvs = [[1], [0]] if swap else [[0], [1]]
    rate = total_correlation(d, rvs, [2])
    assert lower_intrinsic_mutual_information(d, rvs, [2]) == pytest.approx(rate, abs=1e-9)
    assert upper_intrinsic_mutual_information(d, rvs, [2]) == pytest.approx(rate, abs=1e-9)


@settings(max_examples=25, deadline=None)
@given(a=crossover, b=crossover)
def test_markov_chain_through_eve(a, b):
    """
    If X - Z - Y the rate is 0, since I[X:Y|Z] = 0.
    """
    d = _bsc_chain(a, b)
    assert upper_intrinsic_mutual_information(d, [[0], [2]], [1]) == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=25, deadline=None)
@given(
    pxy=st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=4, max_size=4),
    pz=st.floats(min_value=0.01, max_value=0.99),
)
def test_independent_eavesdropper(pxy, pz):
    """
    If Z is independent of XY the rate is I[X:Y].
    """
    pxy = np.array(pxy) / sum(pxy)
    outcomes = [f"{x}{y}{z}" for x, y, z in np.ndindex(2, 2, 2)]
    pmf = [pxy[2 * x + y] * (pz if z else 1 - pz) for x, y, z in np.ndindex(2, 2, 2)]
    d = Distribution(outcomes, pmf)
    rate = total_correlation(d, [[0], [1]])
    assert lower_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)
    assert upper_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)


@settings(max_examples=25, deadline=None)
@given(p=crossover)
def test_symmetric_xor(p):
    """
    X uniform, Y = X xor N, Z = N: Z is independent of X, so I[X:Y] - I[X:Z]
    meets the upper bound I[X:Y] = 1 - h(p).
    """
    d = Distribution(["000", "011", "101", "110"], [(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])
    rate = total_correlation(d, [[0], [1]])
    assert lower_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)
    assert upper_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)


@pytest.mark.flaky(reruns=5)
def test_chitambar_bob_speaks():
    """
    Bob achieves I[X:Y|Z] = 1/3 by one-way communication; Alice cannot.
    """
    d = chitambar_bob_speaks
    rate = d.secret_rate
    assert upper_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)
    assert one_way_skar(d, [1], [0], [2]) == pytest.approx(rate, abs=1e-4)
    assert one_way_skar(d, [0], [1], [2]) < rate - 1e-2
    assert necessary_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_chitambar_two_way():
    """
    I[X:Y|Z] = 1/5 needs both parties to speak: one-way communication in
    either direction falls short, while iterated discarding achieves it.
    """
    d = chitambar_two_way
    rate = d.secret_rate
    assert upper_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)
    assert iterated_discarding_skar(d, [[0], [1]], [2], rng=0) == pytest.approx(rate, abs=1e-4)
    assert necessary_intrinsic_mutual_information(d, [[0], [1]], [2]) < rate - 1e-2


@pytest.mark.flaky(reruns=5)
def test_james_problem():
    """
    The one-way rate from Alice meets the upper bound I[X:Y] = 1/2.
    """
    d = james_problem
    rate = d.secret_rate
    assert upper_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-9)
    assert necessary_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(rate, abs=1e-4)


@settings(max_examples=10, deadline=None)
@given(a=crossover, b=crossover, c=crossover, e=crossover)
def test_reversely_degraded(a, b, c, e):
    """
    The rate is I[X:Y|Z], the sum of the two components' rates.
    """
    d = reversely_degraded(a, b, c, e)
    assert total_correlation(d, [[0], [1]], [2]) == pytest.approx(d.secret_rate, abs=1e-9)
    assert upper_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(d.secret_rate, abs=1e-9)


@pytest.mark.parametrize("alpha", [2.0, 2.5, 3.0])
def test_gisin_wolf_zero(alpha):
    """
    For 2 <= alpha <= 3, Eve can degrade Z so that X and Y are independent:
    send the erasure symbol, and the fraction 2 / w of each pair symbol of
    weight w, to a common output.
    """
    d = gisin_wolf(alpha)
    assert d.secret_rate == 0.0
    outcomes, pmf = [], []
    for (x, y, z), p in zip(d.outcomes, d.pmf, strict=True):
        share = 1.0 if z == "6" else 2 / (21 * p)
        outcomes.append((x, y, "*"))
        pmf.append(share * p)
        if share < 1:
            outcomes.append((x, y, z))
            pmf.append((1 - share) * p)
    degraded = Distribution(outcomes, pmf)
    assert total_correlation(degraded, [[0], [1]], [2]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.5, 4.0, 5.0])
def test_gisin_wolf_unknown(alpha):
    """
    Outside [2, 3] the rate is positive but unknown.
    """
    assert not hasattr(gisin_wolf(alpha), "secret_rate")


_sources = [
    intrinsic_1,
    intrinsic_2,
    intrinsic_3,
    bound_information,
    deterministic_erasure_source,
    chitambar_bob_speaks,
    chitambar_two_way,
    james_problem,
    reversely_degraded(0.1, 0.2, 0.15, 0.1),
    gisin_wolf(2.5),
]

_lower_bounds = [
    lower_intrinsic_mutual_information,
    necessary_intrinsic_mutual_information,
    partial(iterated_discarding_skar, rng=0),
]

# Fewer restarts can only raise a minimization, so the check stays valid.
_upper_bounds = [
    upper_intrinsic_mutual_information,
    intrinsic_mutual_information,
    partial(less_noisy_intrinsic_mutual_information, niter=2, rng=0),
]


@pytest.mark.parametrize("dist", _sources)
@pytest.mark.parametrize("bound", _lower_bounds)
def test_lower_bounds_below_rate(dist, bound):
    """
    No lower bound exceeds a proven rate.
    """
    assert bound(dist, [[0], [1]], [2]) <= dist.secret_rate + 1e-6


@pytest.mark.parametrize("dist", _sources)
@pytest.mark.parametrize("bound", _upper_bounds)
def test_upper_bounds_above_rate(dist, bound):
    """
    No upper bound falls below a proven rate.
    """
    assert bound(dist, [[0], [1]], [2]) >= dist.secret_rate - 1e-6
