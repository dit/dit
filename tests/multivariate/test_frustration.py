"""
Tests for dit.multivariate.frustration.
"""

from itertools import permutations, product

import numpy as np
import pytest
from hypothesis import given, settings
from scipy.optimize import brentq

from dit import Distribution
from dit.algorithms import maxent_dist
from dit.example_dists import Xor, giant_bit
from dit.exceptions import ditException
from dit.multivariate import coinformation, frustration, pairwise_frustration
from dit.multivariate.frustration import _branch_and_bound, _circuits, _LogLinear, _maxent, _ordered_start
from dit.utils.testing import distributions

PAIRS = [[0, 1], [0, 2], [1, 2]]
W = Distribution(["001", "010", "100"], [1 / 3] * 3)
HUB = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 4)]


def dense(d):
    d = d.copy()
    d.make_dense()
    return d.pmf.reshape([len(a) for a in d.alphabet])


def from_array(p):
    outcomes = ["".join(map(str, x)) for x in product(*[range(k) for k in p.shape])]
    return Distribution(outcomes, list(p.ravel() / p.sum()))


def ternary_potts(J):
    """The three-state Potts triangle with coupling J and no field."""
    x = np.arange(3)
    eq = lambda a, b: (a == b).astype(float)
    E = J * (
        eq(x[:, None, None], x[None, :, None])
        + eq(x[:, None, None], x[None, None, :])
        + eq(x[None, :, None], x[None, None, :])
    )
    return from_array(np.exp(E))


def xorsat(n, clauses, rhs, beta=1.0):
    """Soft parity constraints: p(x) proportional to exp(beta * #satisfied clauses)."""
    p = np.zeros((2,) * n)
    for x in product((0, 1), repeat=n):
        sat = sum((sum(x[i] for i in C) % 2) == b for C, b in zip(clauses, rhs, strict=True))
        p[x] = np.exp(beta * sat)
    return from_array(p)


def test_w():
    """W is a maximally frustrated hard-core triangle: the weakest bond costs 2/3 bit."""
    assert pairwise_frustration(W) == pytest.approx(2 / 3, abs=1e-6)
    assert frustration(W, order=2) == pytest.approx(2 / 3, abs=1e-6)


def test_and_is_collider():
    """For a collider the cut pair is independent, so F equals the negative co-information."""
    d = Distribution(["000", "010", "100", "111"], [1 / 4] * 4)
    assert pairwise_frustration(d) == pytest.approx(-coinformation(d), abs=1e-6)


@pytest.mark.parametrize("dist", [Xor(), giant_bit(3, 2), Distribution(["000", "001", "010", "011"], [1 / 4] * 4)])
def test_unfrustrated(dist):
    """Third-order synergy, redundancy and independence carry no frustration."""
    assert pairwise_frustration(dist) == pytest.approx(0.0, abs=1e-6)


def test_two_variables():
    """A single pair has no loop."""
    assert pairwise_frustration(W, [[0], [1]]) == 0.0


def test_masked_by_redundancy():
    """W alongside a giant bit with zero total co-information keeps its 2/3 bit."""
    h = lambda b: -b * np.log2(b) - (1 - b) * np.log2(1 - b)
    b = brentq(lambda b: h(b) - (2 - np.log2(3)), 1e-6, 0.5)
    d = W @ Distribution(["000", "111"], [1 - b, b])
    rvs = [[0, 3], [1, 4], [2, 5]]
    assert coinformation(d, rvs) == pytest.approx(0.0, abs=1e-9)
    assert pairwise_frustration(d, rvs) == pytest.approx(2 / 3, abs=1e-6)


def test_three_colouring():
    """Proper colourings satisfy every bond, but are frustrated in the information sense."""
    d = Distribution(["".join(map(str, s)) for s in permutations(range(3))], [1 / 6] * 6)
    assert pairwise_frustration(d) == pytest.approx(1.0, abs=1e-6)


def test_relabeling_invariance():
    """Relabeling the letters of one variable changes nothing."""
    d = Distribution(["002", "020", "101", "110", "212", "221"], [1 / 6] * 6)
    assert pairwise_frustration(d) == pytest.approx(1.0, abs=1e-6)


def test_one_hot_four():
    """The one-hot distribution on four bits is a frustrated K4."""
    d = Distribution(["0001", "0010", "0100", "1000"], [1 / 4] * 4)
    assert pairwise_frustration(d) == pytest.approx(0.771553, abs=1e-5)


def test_node_budget_warns():
    """An exhausted node budget returns the best value found, with a warning."""
    d = ternary_potts(1.0)
    with pytest.warns(UserWarning, match="did not certify"):
        value = pairwise_frustration(d, max_nodes=2)
    assert value >= 0.064755 - 1e-5


@pytest.mark.slow
def test_potts_ferromagnet():
    """The Potts ferromagnet hides a frustrated binary triangle."""
    assert pairwise_frustration(ternary_potts(1.0)) == pytest.approx(0.064755, abs=1e-5)


def closed_form(d):
    """The three-bit closed form computed from dit's own pairwise maxent distribution."""
    m = maxent_dist(d, PAIRS)
    p = np.clip(dense(m), 1e-300, None)
    lors = []
    for i, j in PAIRS:
        k = 3 - i - j
        idx = lambda a, b, i=i, j=j, k=k: tuple({i: a, j: b, k: 0}[t] for t in range(3))
        lors.append(np.log(p[idx(1, 1)] * p[idx(0, 0)] / (p[idx(1, 0)] * p[idx(0, 1)])))
    loop = np.prod(lors)
    cmi = min(coinformation(m, [[i], [j]], [3 - i - j]) for i, j in PAIRS)
    return loop, cmi, coinformation(m)


@settings(deadline=None, max_examples=30)
@given(dist=distributions(alphabets=(2,) * 3, nondegenerate=True, zeros=False))
def test_binary_closed_form(dist):
    """On three bits F is the weakest-bond CMI exactly when the coupling loop is negative."""
    # keep away from the simplex boundary, where the reference maxent_dist loses accuracy
    p = 0.9 * dense(dist) + 0.1 / 8
    dist = from_array(p)
    loop, cmi, coi = closed_form(dist)
    value = pairwise_frustration(dist)
    if abs(loop) > 1e-3:
        assert value == pytest.approx(cmi if loop < 0 else 0.0, abs=1e-5)
    assert value >= -coi - 1e-5
    assert value <= cmi + 1e-5


@settings(deadline=None, max_examples=10)
@given(dist=distributions(alphabets=(2,) * 3))
def test_nonnegative(dist):
    """F is a divergence."""
    assert pairwise_frustration(dist) >= 0.0


@settings(deadline=None, max_examples=10)
@given(dist=distributions(alphabets=(2,) * 4))
def test_order_three_on_four_variables(dist):
    """The triples of four variables contain no cycle, so three-way frustration vanishes."""
    assert frustration(dist, order=3) == 0.0


def test_order_at_least_n():
    """With no proper k-subsets there is nothing to frustrate."""
    assert frustration(W, order=3) == 0.0


@pytest.mark.parametrize("order", [0, 1])
def test_bad_order(order):
    """The order must be at least 2."""
    with pytest.raises(ditException, match="order must be at least 2"):
        frustration(W, order=order)


def test_inconsistent_xorsat_is_frustrated():
    """Four parity constraints whose right-hand sides sum to 1 cannot all hold."""
    assert frustration(xorsat(5, HUB, [0, 0, 0, 1]), order=3) == pytest.approx(0.107287, abs=1e-5)


def test_consistent_xorsat_is_not():
    """A satisfiable system of parity constraints is unfrustrated."""
    assert frustration(xorsat(5, HUB, [0, 0, 0, 0]), order=3) == pytest.approx(0.0, abs=1e-6)


def test_xorsat_gauge_invariance():
    """Flipping the right-hand sides of every clause through variable 1 relabels X1."""
    a = frustration(xorsat(5, HUB, [0, 0, 0, 1]), order=3)
    b = frustration(xorsat(5, HUB, [1, 1, 0, 1]), order=3)
    assert a == pytest.approx(b, abs=1e-6)


def test_six_variable_cycle():
    """A four-triple cycle on six variables, with a greedy choice of basis."""
    d = xorsat(6, [(0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5)], [0, 0, 0, 1])
    assert frustration(d, order=3) == pytest.approx(0.151073, abs=1e-5)


def test_cycle_space_too_large():
    """Pairwise on eight variables has a cycle space of dimension 21."""
    d = Distribution(["".join(x) for x in product("01", repeat=8)], [1 / 256] * 256)
    with pytest.raises(ditException, match="too large to enumerate"):
        frustration(d)


def test_seriation_start_matches_exhaustive():
    """The seriation heuristic for large alphabets finds the best ordered start here."""
    p = dense(ternary_potts(1.0))
    fam = _LogLinear(p.shape, 2)
    exhaustive = _ordered_start(fam, p)
    heuristic = _ordered_start(fam, p, max_exhaustive=0)
    kl = lambda q: float((p * np.log2(p / q)).sum())
    assert kl(heuristic) == pytest.approx(kl(exhaustive), abs=1e-6)


def test_branch_and_bound_without_incumbent():
    """Without an incumbent the branch-and-bound certifies the optimum on its own."""
    p = dense(Distribution(["".join(map(str, s)) for s in permutations(range(3))], [1 / 6] * 6))
    pk = _maxent(p, 2)
    fam = _LogLinear(pk.shape, 2)
    F, cons = fam.constraints(_circuits(3, 2))
    value, lower = _branch_and_bound(fam, pk, F, cons, np.inf, max_nodes=20000, tol=1e-7)
    assert value == pytest.approx(1.0, abs=1e-6)
    assert lower == pytest.approx(value, abs=1e-6)


def test_order_four_needs_five_variables():
    """The quadruples of five variables form one cycle; an odd number of odd clauses frustrates it."""
    quads = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4)]
    assert frustration(xorsat(5, quads, [0, 0, 0, 0, 1]), order=4) == pytest.approx(0.067318, abs=1e-5)
    assert frustration(xorsat(5, quads, [0, 0, 0, 0, 0]), order=4) == pytest.approx(0.0, abs=1e-6)
