"""
Tests for dit.multivariate.higher_order_common_information.
"""

from itertools import combinations, product
from math import prod

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.example_dists import Xor, dyadic, giant_bit, noisy_copies, triadic
from dit.multivariate import gk_common_information
from dit.multivariate import higher_order_common_information as R
from dit.rate_distortion import information_bottleneck_rate
from dit.shannon import mutual_information
from dit.utils.testing import distributions


def _h(p):
    """
    The binary entropy.
    """
    return 0.0 if p <= 0 or p >= 1 else float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def _noisy_copies_hci(qs):
    """
    The higher-order common information of X_j = U xor N_j, N_j ~ Bernoulli(q_j)
    (Theorem 4 of :cite:`ostergaard2024higher`).
    """
    n = len(qs)
    cs = [1 - 2 * q for q in qs]
    values = []
    for i in range(n):
        d = sorted(c for j, c in enumerate(cs) if j != i)
        beta = d[0] * cs[i] ** (2 ** (n - 2)) * prod(d[m - 1] ** (2 ** (m - 2)) for m in range(2, n))
        values.append(1 - _h((1 - beta) / 2))
    return min(values)


def _aggregate(outcomes):
    """
    The distribution of equally likely (possibly repeated) outcomes.
    """
    pmf = {}
    for outcome in outcomes:
        pmf[outcome] = pmf.get(outcome, 0) + 1 / len(outcomes)
    return Distribution(list(pmf), list(pmf.values()))


def _erasure_example():
    """
    X_0 in {0, 1, 2} with p = (1/2, 1/4, 1/4), and X_1 = X_2 an erasure of X_0
    that keeps it with probability 3/5.
    """
    outcomes, pmf = [], []
    for x, p in zip("012", (1 / 2, 1 / 4, 1 / 4), strict=True):
        outcomes += [x + x + x, x + "ee"]
        pmf += [p * 3 / 5, p * 2 / 5]
    return Distribution(outcomes, pmf)


def _and_gates(coarsen):
    """
    (A & B, A & C, B & C) for independent fair bits, or (A & B, A & C, (B, C)).
    """
    return _aggregate(
        [(str(a & b), str(a & c), str(b & c) if coarsen else f"{b}{c}") for a, b, c in product((0, 1), repeat=3)]
    )


def test_one_variable():
    """
    A single variable shares its entropy with itself.
    """
    d = Distribution(["0", "1", "2"], [1 / 2, 1 / 4, 1 / 4])
    assert R(d) == pytest.approx(1.5)


def test_two_variables():
    """
    For two variables it is the mutual information.
    """
    d = noisy_copies(0.1, 2)
    assert R(d) == pytest.approx(mutual_information(d, [0], [1]))


def test_groups():
    """
    Variables may be groups of indices, and a subset of them.
    """
    d = noisy_copies(0.1, 3)
    assert R(d, rvs=[[0], [1]]) == pytest.approx(mutual_information(d, [0], [1]))
    assert R(d, rvs=[[0, 1], [2]]) == pytest.approx(mutual_information(d, [0, 1], [2]))


@pytest.mark.parametrize(("dist", "value"), [(giant_bit(3, 2), 1.0), (giant_bit(3, 3), np.log2(3)), (triadic, 1.0)])
def test_gacs_korner_meets_pairwise(dist, value):
    """
    When the Gacs-Korner common information equals the least pairwise mutual
    information, both bound it from either side.
    """
    assert R(dist) == pytest.approx(value)


def test_pairwise_independent():
    """
    It vanishes when a pair of variables is independent.
    """
    assert R(Xor()) == pytest.approx(0.0, abs=1e-9)


def test_dyadic():
    """
    Each pair of dyadic variables shares a different bit, and nothing is shared
    by all three, so it vanishes although every pair shares a bit (Example 4 of
    :cite:`ostergaard2024higher`).
    """
    assert R(dyadic) == pytest.approx(0.0, abs=1e-9)


def test_independent_components():
    """
    For variables made of independent components, it is the entropy of the
    components common to all of them (Lemma 8 of :cite:`ostergaard2024higher`).
    """
    d = _aggregate([(g + a + b, g + a + c, g + b + c) for g, a, b, c in product("01", repeat=4)])
    assert R(d) == pytest.approx(1.0)


@pytest.mark.parametrize("q", [0.02, 0.1, 0.25])
def test_noisy_copies(q):
    """
    Three noisy copies of a fair bit, with crossover q, have
    1 - h((1 - (1 - 2q)^4) / 2) (Corollary 4 of :cite:`ostergaard2024higher`).
    """
    expected = 1 - _h((1 - (1 - 2 * q) ** 4) / 2)
    assert expected == pytest.approx(_noisy_copies_hci([q] * 3))
    assert R(noisy_copies(q, 3), rng=0) == pytest.approx(expected, abs=1e-4)


def test_noisy_copies_heterogeneous():
    """
    Copies with different crossovers (Theorem 4 of :cite:`ostergaard2024higher`).
    """
    qs = [0.05, 0.1, 0.2]
    assert R(noisy_copies(qs), rng=0) == pytest.approx(_noisy_copies_hci(qs), abs=1e-4)


def test_symmetric():
    """
    The order of the variables does not matter.
    """
    d = noisy_copies([0.05, 0.1, 0.2])
    assert R(d, rvs=[[2], [0], [1]], rng=0) == pytest.approx(R(d, rng=0), abs=1e-4)


def test_erasure_example():
    """
    The stage optimizer must be stochastic: erasing X_0 with probability 2/5
    gives alpha^2 H[X_0] = 27/50, more than any deterministic function of X_0
    (Example 3 of :cite:`ostergaard2024higher`). Duplicating X_1 lowers the value
    from I[X_0 : X_1] = 9/10: it is not invariant under adding a copy.
    """
    d = _erasure_example()
    assert R(d, rng=0) == pytest.approx(27 / 50, abs=1e-4)
    assert R(d, rvs=[[0], [1]]) == pytest.approx(9 / 10)


def test_selector():
    """
    X_0 = (A, B) and X_1 = X_2 = (Q, A if Q = 0 else B): every representation of
    X_0 within budget shares at most half a bit with X_1 (Lemma 2 of
    :cite:`ostergaard2024higher`).
    """
    outcomes = []
    for a, b, q in product("01", repeat=3):
        y = a if q == "0" else b
        outcomes.append((a + b, q + y, q + y))
    assert R(_aggregate(outcomes), rng=0) == pytest.approx(0.5, abs=1e-4)


def test_local_processing_can_increase():
    """
    Coarse-graining (B, C) to B & C raises the value from zero: it does not
    satisfy the data processing inequality. With reference (B, C), every stage
    optimizer is a degraded B or C, which is independent of the remaining AND.
    After coarse-graining, by symmetry, it is the information bottleneck of
    A & B about A & C at their mutual information.
    """
    assert R(_and_gates(coarsen=False), rng=0) == pytest.approx(0.0, abs=1e-3)
    d = _and_gates(coarsen=True)
    pair = d.marginal([0, 1])
    expected = information_bottleneck_rate(pair, mutual_information(pair, [0], [1]), rng=0)
    assert expected > 0.008
    assert R(d, rng=0) == pytest.approx(expected, abs=1e-4)


@settings(max_examples=5, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_bounds(dist):
    """
    It lies between the Gacs-Korner common information and the least pairwise
    mutual information.
    """
    value = R(dist, rng=0)
    lower = gk_common_information(dist)
    upper = min(mutual_information(dist, [i], [j]) for i, j in combinations(range(3), 2))
    assert lower - 1e-6 <= value <= upper + 1e-6
