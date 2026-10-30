"""
Tests for dit.example_dists.common_information
"""

from itertools import combinations, permutations, product

import pytest

from dit.example_dists import noisy_copies, symmetric_common_information_witness
from dit.example_dists.common_information import _TUNED_Q
from dit.multivariate import caekl_mutual_information as J
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import entropy as H
from dit.multivariate import exact_common_information as G
from dit.multivariate import functional_common_information as F
from dit.multivariate import gk_common_information as K
from dit.multivariate import mss_common_information as M
from dit.multivariate import wyner_common_information as C


def _is_exchangeable(d):
    """
    Whether the joint distribution is invariant under every permutation of its
    variables.
    """
    n = d.outcome_length()
    table = {tuple(o): p for o, p in d.zipped()}
    for perm in permutations(range(n)):
        for outcome, p in table.items():
            permuted = tuple(outcome[i] for i in perm)
            if abs(table.get(permuted, 0.0) - p) > 1e-12:
                return False
    return True


# ---------------------------------------------------------------------------
# noisy_copies
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workhorse():
    """
    The small three-copy noisy source used to separate the middle of the chain.
    """
    return noisy_copies(0.1, 3)


def test_noisy_normalized(workhorse):
    """
    The pmf sums to one.
    """
    assert sum(workhorse.pmf) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_noisy_support(n):
    """
    n binary variables with 2**n outcomes.
    """
    d = noisy_copies(0.1, n)
    assert d.outcome_length() == n
    assert len(d.outcomes) == 2**n
    assert set(d.alphabet[0]) == {"0", "1"}


@pytest.mark.parametrize("n", [2, 3, 4])
def test_noisy_exchangeable(n):
    """
    The noisy source is exchangeable.
    """
    assert _is_exchangeable(noisy_copies(0.1, n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_noisy_pmf(n):
    """
    The probability of an outcome depends only on its Hamming weight,
    p_k = (q^k (1-q)^(n-k) + q^(n-k) (1-q)^k) / 2.
    """
    q = 0.1
    d = noisy_copies(q, n)
    table = {tuple(o): p for o, p in d.zipped()}
    for bits in product("01", repeat=n):
        k = sum(b == "1" for b in bits)
        expected = (q**k * (1 - q) ** (n - k) + q ** (n - k) * (1 - q) ** k) / 2
        assert table[bits] == pytest.approx(expected)


def test_noisy_crossover_matches_dsbs():
    """
    For n = 2 the pairwise crossover P(X_0 != X_1) is 2 q (1 - q), Wyner's doubly
    symmetric binary source.
    """
    q = 0.1
    d = noisy_copies(q, 2)
    table = {tuple(o): p for o, p in d.zipped()}
    disagree = table[("0", "1")] + table[("1", "0")]
    assert disagree == pytest.approx(2 * q * (1 - q))


def test_noisy_no_shared_or_private(workhorse):
    """
    The noisy source has no deterministic shared structure (K = 0) and no private
    randomness (M = H).
    """
    assert float(K(workhorse)) == pytest.approx(0.0, abs=1e-9)
    assert float(M(workhorse)) == pytest.approx(float(H(workhorse)))


def test_noisy_heterogeneous():
    """
    With a crossover q_i per copy, copies i and j disagree with probability
    q_i (1 - q_j) + q_j (1 - q_i), and the number of copies is the number of
    crossovers.
    """
    qs = [0.05, 0.1, 0.2]
    d = noisy_copies(qs)
    assert d.outcome_length() == 3
    table = {tuple(o): p for o, p in d.zipped()}
    for i, j in combinations(range(3), 2):
        disagree = sum(p for o, p in table.items() if o[i] != o[j])
        assert disagree == pytest.approx(qs[i] * (1 - qs[j]) + qs[j] * (1 - qs[i]))


def test_noisy_heterogeneous_matches_scalar():
    """
    Equal per-copy crossovers give the exchangeable source.
    """
    assert noisy_copies([0.1] * 4).is_approx_equal(noisy_copies(0.1, 4))


def test_noisy_heterogeneous_bad_n():
    """
    ``n`` must agree with the number of crossovers.
    """
    with pytest.raises(ValueError, match="number of crossovers"):
        noisy_copies([0.1, 0.2], n=3)


# ---------------------------------------------------------------------------
# symmetric_common_information_witness -- structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_witness_structure(n):
    """
    n variables over a common eight-symbol alphabet with 2**(2n+1) outcomes,
    exchangeable and normalized.
    """
    d = symmetric_common_information_witness(n)
    assert d.outcome_length() == n
    assert len(d.alphabet[0]) == 8
    assert len(d.outcomes) == 2 ** (2 * n + 1)
    assert sum(d.pmf) == pytest.approx(1.0)
    assert _is_exchangeable(d)


def test_witness_default_q():
    """
    A ``None`` crossover uses the per-n tuned value.
    """
    auto = symmetric_common_information_witness(3)
    explicit = symmetric_common_information_witness(3, _TUNED_Q[3])
    assert auto.is_approx_equal(explicit)


def test_witness_requires_two_variables():
    """
    Fewer than two variables is rejected.
    """
    with pytest.raises(ValueError):
        symmetric_common_information_witness(1)


# ---------------------------------------------------------------------------
# the common-information ordering
# ---------------------------------------------------------------------------


# The full chain 0 < K < J < B < C < G < F < M < H is certified in two cheap
# pieces that together exercise every exported distribution and every rung: the
# small noisy workhorse separates the Wyner/exact-heavy middle B < C < G < F, and
# the assembled witness lifts the endpoints (0 < K via the giant bit, F < M < H via
# the private bits) while reconfirming K < J < B. Running the expensive Wyner/exact
# optimizers on the 128-outcome witness directly is both slow and bound-sensitive,
# so those rungs are tested on the eight-outcome workhorse instead.


@pytest.mark.flaky(reruns=5)
def test_workhorse_separates_middle(workhorse):
    """
    The noisy source strictly separates the middle of the ordering,
    0 = K < J < B < C < G < F, with M = H (no private randomness).
    """
    k = float(K(workhorse))
    j = float(J(workhorse))
    b = float(B(workhorse))
    c = float(C(workhorse, bound=6))
    g = float(G(workhorse, bound=6))
    f = float(F(workhorse))
    m = float(M(workhorse))
    h = float(H(workhorse))
    assert k == pytest.approx(0.0, abs=1e-9)
    assert k + 1e-3 < j < b < c - 1e-3
    assert c + 1e-3 < g < f - 1e-3
    assert m == pytest.approx(h)


def test_witness_lifts_endpoints():
    """
    On the assembled witness the giant bit lifts K to one bit (0 < K) and the
    private bits open M < H; K < J < B stays strict.
    """
    d = symmetric_common_information_witness(3)
    k = float(K(d))
    j = float(J(d))
    b = float(B(d))
    f = float(F(d))
    m = float(M(d))
    h = float(H(d))
    assert k == pytest.approx(1.0, abs=1e-6)
    assert k < j < b
    assert f < m - 1e-6
    assert m < h - 1e-6
