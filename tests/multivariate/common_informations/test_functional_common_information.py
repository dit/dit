"""
Tests for dit.multivariate.functional_common_information.
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution, random_distribution
from dit.distconst import RVFunctions, insert_rvf, modify_outcomes
from dit.example_dists import dyadic, giant_bit, triadic
from dit.helpers import normalize_rvs, parse_rvs
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import functional_common_information as F
from dit.multivariate.common_informations._functional_partition import (
    conditional_dtc,
    labels_from_partition,
    partition_entropy,
    prepare_functional_search,
)
from dit.multivariate.common_informations.functional_common_information import functional_markov_chain
from dit.multivariate.entropy import entropy
from dit.utils import partitions


def _reference_partition_metrics(dist, partition, rvs, crvs):
    """Cross-check helpers against insert_rvf + dit multivariate measures."""
    rv_names = dist.get_rv_names()
    dist = modify_outcomes(dist.copy(), tuple)
    if rv_names is not None:
        dist.set_rv_names(rv_names)
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    rvs = [list(parse_rvs(dist, rv)[1]) for rv in rvs]
    crvs = list(parse_rvs(dist, crvs)[1])

    bf = RVFunctions(dist)
    w_idx = dist.outcome_length()
    d = insert_rvf(dist, bf.from_partition(partition))
    h_ref = entropy(d, [w_idx])
    b_ref = B(d, rvs, crvs + [w_idx])
    return h_ref, b_ref


@pytest.mark.parametrize(
    "partition",
    [
        frozenset(
            {
                frozenset([("0", "0", "0")]),
                frozenset([("0", "1", "1")]),
                frozenset([("1", "0", "1")]),
                frozenset([("1", "1", "0")]),
            }
        ),
        frozenset({frozenset([("0", "0", "0"), ("1", "1", "0")]), frozenset([("0", "1", "1"), ("1", "0", "1")])}),
        frozenset({frozenset([("0", "0", "0"), ("0", "1", "1"), ("1", "0", "1"), ("1", "1", "0")])}),
    ],
)
def test_partition_helpers_match_insert_rvf(partition):
    """
    partition_entropy and conditional_dtc agree with Distribution construction.
    """
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    ctx = prepare_functional_search(d)
    pmf_size = int(np.prod(ctx.shape))
    labels = labels_from_partition(partition, ctx.outcome_to_flat, pmf_size)

    h_fast = partition_entropy(ctx.pmf, labels)
    b_fast = conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs)
    h_ref, b_ref = _reference_partition_metrics(d, partition, None, None)

    assert h_fast == pytest.approx(h_ref)
    assert b_fast == pytest.approx(b_ref)


def test_fci1():
    """
    Test known values.
    """
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    assert F(d) == pytest.approx(2.0)
    assert F(d, [[0], [1]]) == pytest.approx(0.0)
    assert F(d, [[0], [1]], [2]) == pytest.approx(1.0)


def test_fci2():
    """
    Test known values w/ rv names.
    """
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    d.set_rv_names("XYZ")
    assert F(d) == pytest.approx(2.0)
    assert F(d, ["X", "Y"]) == pytest.approx(0.0)
    assert F(d, ["X", "Y"], "Z") == pytest.approx(1.0)


def test_fci3():
    """
    Test against known values
    """
    outcomes = [
        "000",
        "a00",
        "00c",
        "a0c",
        "011",
        "a11",
        "101",
        "b01",
        "01d",
        "a1d",
        "10d",
        "b0d",
        "110",
        "b10",
        "11c",
        "b1c",
    ]
    pmf = [1 / 16] * 16
    d = Distribution(outcomes, pmf)
    assert F(d) == pytest.approx(2.0)


MDBSI16 = Distribution(
    [
        "000",
        "a00",
        "00c",
        "a0c",
        "011",
        "a11",
        "101",
        "b01",
        "01d",
        "a1d",
        "10d",
        "b0d",
        "110",
        "b10",
        "11c",
        "b1c",
    ],
    [1 / 16] * 16,
)


_PARITY = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)


def _random_dist(seed, n_outcomes):
    rng = np.random.default_rng(seed)
    outcomes = ["".join(t) for t in product("01", repeat=3)]
    chosen = list(rng.choice(outcomes, size=n_outcomes, replace=False))
    pmf = rng.integers(1, 4, size=n_outcomes).astype(float) if seed % 2 else rng.dirichlet(np.ones(n_outcomes))
    return Distribution(chosen, pmf / pmf.sum())


def _brute_force(dist, rvs, crvs):
    """Minimum H(W) over every partition of the support with B(rvs | crvs, W) = 0."""
    ctx = prepare_functional_search(dist, rvs, crvs)
    size = int(np.prod(ctx.shape))
    best = np.inf
    for part in partitions(ctx.dist.outcomes):
        labels = labels_from_partition(frozenset(frozenset(b) for b in part), ctx.outcome_to_flat, size)
        if np.isclose(conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs), 0):
            best = min(best, partition_entropy(ctx.pmf, labels))
    return best


def _assert_valid_partition(dist, rvs, crvs, h, partition):
    ctx = prepare_functional_search(dist, rvs, crvs)
    labels = labels_from_partition(partition, ctx.outcome_to_flat, int(np.prod(ctx.shape)))
    assert np.isclose(conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs), 0)
    assert partition_entropy(ctx.pmf, labels) == pytest.approx(h)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(("rvs", "crvs"), [([[0], [1], [2]], []), ([[0], [1]], [2])])
def test_matches_brute_force(seed, rvs, crvs):
    """The exact cover agrees with exhaustive search, with and without conditioning."""
    dist = _random_dist(seed, 4 + seed % 3)
    stats: dict = {}
    h = functional_markov_chain(dist, rvs, crvs, _stats=stats)
    assert h == pytest.approx(_brute_force(dist, rvs, crvs))
    _assert_valid_partition(dist, rvs, crvs, h, stats["partition"])


@pytest.mark.parametrize("dist", [_PARITY, MDBSI16, giant_bit(3, 3), dyadic, triadic])
def test_returns_optimal_partition(dist):
    """_stats['partition'] is a valid functional Markov variable achieving H(W)."""
    stats: dict = {}
    h = functional_markov_chain(dist, _stats=stats)
    _assert_valid_partition(dist, None, None, h, stats["partition"])


def test_product_component_is_a_single_block():
    """Independent variables over large alphabets need no rectangle enumeration."""
    d = Distribution(list(product(range(6), repeat=4)), [1 / 6**4] * 6**4)
    assert F(d) == pytest.approx(0.0)


def test_random_three_by_three():
    """A generic 27-outcome distribution (intractable for partition search) solves quickly."""
    d = random_distribution(3, 3, prng=np.random.RandomState(0))
    stats: dict = {}
    h = functional_markov_chain(d, _stats=stats)
    _assert_valid_partition(d, None, None, h, stats["partition"])
    assert B(d) - 1e-9 <= h <= entropy(d) + 1e-9
