"""
Tests for dit.multivariate.pairwise_frustration.
"""

from itertools import permutations, product

import numpy as np
import pytest
from hypothesis import given, settings
from scipy.optimize import brentq

from dit import Distribution
from dit.algorithms import maxent_dist
from dit.example_dists import Xor, giant_bit
from dit.multivariate import coinformation, pairwise_frustration
from dit.utils.testing import distributions

PAIRS = [[0, 1], [0, 2], [1, 2]]
W = Distribution(["001", "010", "100"], [1 / 3] * 3)


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


def test_w():
    """W is a maximally frustrated hard-core triangle: the weakest bond costs 2/3 bit."""
    assert pairwise_frustration(W) == pytest.approx(2 / 3, abs=1e-6)


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
