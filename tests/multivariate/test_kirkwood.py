"""
Tests for dit.multivariate.kirkwood.
"""

from itertools import combinations, product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.algorithms import maxent_dist
from dit.divergences import kullback_leibler_divergence
from dit.example_dists import Xor, dyadic, giant_bit, n_mod_m, triadic
from dit.exceptions import ditException
from dit.multivariate import coinformation, kirkwood_mutual_information, ouroboros_mutual_information
from dit.shannon import mutual_information
from dit.utils.testing import distributions


def weighted(n):
    """A fixed full-support distribution over n bits."""
    outcomes = ["".join(o) for o in product("01", repeat=n)]
    weights = np.arange(1, 2**n + 1, dtype=float) ** 1.5
    return Distribution(outcomes, list(weights / weights.sum()))


def log_kirkwood_normalizer(d):
    """log2 of the sum of the unnormalized Kirkwood approximation."""
    n = d.outcome_length()
    d = d.copy()
    d.make_dense()
    p = d.pmf.reshape([len(a) for a in d.alphabet])
    terms = []
    for x in product(*[range(s) for s in p.shape]):
        log_value = 0.0
        for k in range(1, n):
            for S in combinations(range(n), k):
                others = tuple(i for i in range(n) if i not in S)
                marginal = p.sum(axis=others)[tuple(x[i] for i in S)]
                if marginal == 0:
                    log_value = -np.inf
                    break
                log_value += (-1) ** (n - 1 - k) * np.log2(marginal)
            if log_value == -np.inf:
                break
        terms.append(log_value)
    return np.logaddexp2.reduce(terms)


@pytest.mark.parametrize("n", range(3, 6))
def test_kirkwood_giant_bit(n):
    """The giant bit has no n-way structure beyond its (n-1)-marginals."""
    assert kirkwood_mutual_information(giant_bit(n, 2)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", range(3, 6))
def test_kirkwood_parity(n):
    """Parity carries exactly one bit of n-way structure."""
    assert kirkwood_mutual_information(n_mod_m(n, 2)) == pytest.approx(1.0)


def test_kirkwood_bivariate_is_mi():
    """For two variables the Kirkwood approximation is the product of marginals."""
    d = weighted(3)
    assert kirkwood_mutual_information(d, [[0, 1], [2]]) == pytest.approx(mutual_information(d, [0, 1], [2]))


def test_kirkwood_single_variable():
    """A single variable has zero Kirkwood mutual information."""
    assert kirkwood_mutual_information(Xor(), [[0]]) == 0.0


def test_kirkwood_conditional():
    """Conditioned on X2, the bivariate Kirkwood MI is I[X0 : X1 | X2]."""
    d = weighted(3)
    expected = coinformation(d, [[0], [1]], [2])
    assert kirkwood_mutual_information(d, [[0], [1]], [2]) == pytest.approx(expected)


def test_kirkwood_names():
    """Random variable names are accepted."""
    d = weighted(3)
    d.set_rv_names("XYZ")
    assert kirkwood_mutual_information(d, ["X", "Y", "Z"]) == pytest.approx(kirkwood_mutual_information(d))


def test_kirkwood_subvariables():
    """Any two variables of xor are independent."""
    assert kirkwood_mutual_information(Xor(), [[0], [1]]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("dist", "value"),
    [
        (dyadic, 0.0),
        (triadic, 1.0),
        (Distribution(["000", "010", "100", "111"], [1 / 4] * 4), 0.18872187554086717),
    ],
)
def test_kirkwood_values(dist, value):
    """Known values on three-variable distributions."""
    assert kirkwood_mutual_information(dist, [[0], [1], [2]]) == pytest.approx(value, abs=1e-9)


@settings(deadline=None, max_examples=25)
@given(dist=distributions(alphabets=(2,) * 3))
def test_kirkwood_nonnegative(dist):
    """The Kirkwood mutual information is a divergence."""
    assert kirkwood_mutual_information(dist) >= -1e-9


@pytest.mark.parametrize("n", [3, 4])
@settings(deadline=None, max_examples=15)
@given(data=distributions(alphabets=(2,) * 4))
def test_kirkwood_coinformation_identity(n, data):
    """K = (-1)^n I[X_0 : ... : X_{n-1}] + log2 Z."""
    d = data.marginal(list(range(n))) if n < 4 else data
    rvs = [[i] for i in range(n)]
    expected = (-1) ** n * coinformation(d, rvs) + log_kirkwood_normalizer(d)
    assert kirkwood_mutual_information(d, rvs) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("n", range(3, 6))
def test_ouroboros_giant_bit(n):
    """Every order of ouroboros approximation is exact on the giant bit."""
    d = giant_bit(n, 2)
    for k in range(1, n - 1):
        assert ouroboros_mutual_information(d, order=k) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", range(3, 6))
def test_ouroboros_parity(n):
    """Every order of ouroboros approximation of parity is uniform."""
    d = n_mod_m(n, 2)
    for k in range(1, n - 1):
        assert ouroboros_mutual_information(d, order=k) == pytest.approx(1.0)


def test_ouroboros_three_variables_is_kirkwood():
    """For three variables the order-1 ouroboros is the Kirkwood approximation."""
    d = weighted(3)
    assert ouroboros_mutual_information(d, order=1) == pytest.approx(kirkwood_mutual_information(d))


def test_ouroboros_default_order():
    """The default order is n - 2."""
    d = weighted(4)
    assert ouroboros_mutual_information(d) == pytest.approx(ouroboros_mutual_information(d, order=2))


def test_ouroboros_conditional():
    """Conditioning on an independent variable changes nothing."""
    d = triadic @ Distribution(["0", "1"], [1 / 2, 1 / 2])
    value = ouroboros_mutual_information(d, order=1, rvs=[[0], [1], [2]], crvs=[3])
    assert value == pytest.approx(ouroboros_mutual_information(triadic, order=1))


@pytest.mark.parametrize("rvs", [[[0], [1]], [[0]]])
def test_ouroboros_too_few_variables(rvs):
    """At least three variables are required."""
    with pytest.raises(ditException, match="at least 3 variables"):
        ouroboros_mutual_information(weighted(3), rvs=rvs)


@pytest.mark.parametrize("order", [0, 3])
def test_ouroboros_bad_order(order):
    """The order must lie in [1, n - 2]."""
    with pytest.raises(ditException, match="order must satisfy"):
        ouroboros_mutual_information(weighted(4), order=order)


@pytest.mark.parametrize("k", [1, 2])
def test_ouroboros_maxent_bound(k):
    """O_k upper bounds the divergence to the (k+1)-marginal maxent distribution."""
    d = weighted(4)
    m = maxent_dist(d, [list(S) for S in combinations(range(4), k + 1)])
    assert ouroboros_mutual_information(d, order=k) >= kullback_leibler_divergence(d, m) - 1e-6


def test_ouroboros_detects_lower_order_structure():
    """Triadic plus an independent bit: only the pairwise ouroboros sees the triad."""
    d = triadic @ Distribution(["0", "1"], [1 / 2, 1 / 2])
    rvs = [[0], [1], [2], [3]]
    assert kirkwood_mutual_information(d, rvs) == pytest.approx(0.0, abs=1e-9)
    assert ouroboros_mutual_information(d, order=1, rvs=rvs) == pytest.approx(1.0)
    assert ouroboros_mutual_information(d, order=2, rvs=rvs) == pytest.approx(0.0, abs=1e-9)


@settings(deadline=None, max_examples=25)
@given(dist=distributions(alphabets=(2,) * 4))
def test_ouroboros_nonnegative(dist):
    """The ouroboros mutual information is a divergence."""
    for k in (1, 2):
        assert ouroboros_mutual_information(dist, order=k) >= -1e-9
