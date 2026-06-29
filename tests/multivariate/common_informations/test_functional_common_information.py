"""
Tests for dit.multivariate.functional_common_information.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.distconst import RVFunctions, insert_rvf, modify_outcomes
from dit.helpers import normalize_rvs, parse_rvs
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import functional_common_information as F
from dit.multivariate import gk_common_information as G
from dit.multivariate import mss_common_information as M
from dit.multivariate.common_informations._functional_partition import (
    conditional_dtc,
    labels_from_partition,
    partition_entropy,
    partition_from_joint_mss,
    partition_from_meet,
    prepare_functional_search,
    refinements_by_binary_split,
)
from dit.multivariate.common_informations.functional_common_information import functional_markov_chain
from dit.multivariate.entropy import entropy


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


@pytest.mark.parametrize(
    "dist",
    [
        Distribution(["000", "011", "101", "110"], [1 / 4] * 4),
        Distribution(
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
        ),
    ],
)
def test_mss_partition_is_valid(dist):
    """Joint-MSS outcome partition satisfies B(rvs | crvs, W) = 0."""
    ctx = prepare_functional_search(dist)
    pmf_size = int(np.prod(ctx.shape))
    mss_part = partition_from_joint_mss(dist)
    labels = labels_from_partition(mss_part, ctx.outcome_to_flat, pmf_size)
    assert conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs) == pytest.approx(0.0)


def test_mss_partition_entropy_matches_mss_common_information():
    """H(W) from the MSS partition agrees with mss_common_information."""
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    ctx = prepare_functional_search(d)
    pmf_size = int(np.prod(ctx.shape))
    mss_part = partition_from_joint_mss(d)
    labels = labels_from_partition(mss_part, ctx.outcome_to_flat, pmf_size)
    assert partition_entropy(ctx.pmf, labels) == pytest.approx(M(d))


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


@pytest.mark.parametrize(
    "dist",
    [
        Distribution(["000", "011", "101", "110"], [1 / 4] * 4),
        MDBSI16,
    ],
)
def test_mss_warmstart_reduces_partition_visits(dist):
    """MSS warm start visits no more partitions than finest-partition seed."""
    stats_mss: dict = {}
    stats_finest: dict = {}
    h_mss = functional_markov_chain(dist, _use_mss_warmstart=True, _strategy="coarsen", _stats=stats_mss)
    h_finest = functional_markov_chain(dist, _use_mss_warmstart=False, _strategy="coarsen", _stats=stats_finest)
    assert h_mss == pytest.approx(h_finest)
    assert stats_mss["mss_warmstart"] is True
    assert stats_mss["visited"] <= stats_finest["visited"]


def test_mss_warmstart_immediate_exit_when_f_equals_m():
    """When F = M, MSS seed exits without exploring coarsenings."""
    stats: dict = {}
    h = functional_markov_chain(MDBSI16, _strategy="coarsen", _stats=stats)
    assert h == pytest.approx(2.0)
    assert stats["mss_warmstart"] is True
    assert stats["visited"] == 0


def test_mss_warmstart_fallback_on_named_partial_rvs():
    """Named partial-rvs MSS insert failure falls back to finest partition."""
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    d.set_rv_names("XYZ")
    stats: dict = {}
    h = functional_markov_chain(d, ["X", "Y"], _strategy="coarsen", _stats=stats)
    assert h == pytest.approx(0.0)
    assert stats["mss_warmstart"] is False


@pytest.mark.parametrize(
    "dist",
    [
        Distribution(["000", "011", "101", "110"], [1 / 4] * 4),
        MDBSI16,
    ],
)
def test_meet_partition_is_coarsest_seed(dist):
    """H(W) from the meet partition agrees with gk_common_information."""
    ctx = prepare_functional_search(dist)
    pmf_size = int(np.prod(ctx.shape))
    meet_part = partition_from_meet(dist)
    labels = labels_from_partition(meet_part, ctx.outcome_to_flat, pmf_size)
    assert partition_entropy(ctx.pmf, labels) == pytest.approx(G(dist))


def _min_h_refinement_search(ctx, meet_part, pmf_size):
    """Exhaustive min H among meet refinements with B = 0 (small supports only)."""
    optimal_h = float("inf")
    queue = [meet_part]
    seen = set()
    while queue:
        part = queue.pop()
        if part in seen:
            continue
        seen.add(part)
        labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)
        if np.isclose(conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs), 0):
            h = partition_entropy(ctx.pmf, labels)
            optimal_h = min(optimal_h, h)
        for refined in refinements_by_binary_split(part):
            if refined not in seen:
                queue.append(refined)
    return optimal_h


def test_meet_partition_refines_to_f():
    """F equals the minimum H over meet refinements with B = 0 on small supports."""
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    ctx = prepare_functional_search(d)
    pmf_size = int(np.prod(ctx.shape))
    meet_part = partition_from_meet(d)
    assert _min_h_refinement_search(ctx, meet_part, pmf_size) == pytest.approx(F(d))


@pytest.mark.parametrize(
    "dist, expected",
    [
        (Distribution(["000", "011", "101", "110"], [1 / 4] * 4), 2.0),
    ],
)
def test_fci_dual_search_agree(dist, expected):
    """Coarsen, refine, bidirectional, and auto strategies agree on known F values."""
    h_coarsen = functional_markov_chain(dist, _strategy="coarsen")
    h_refine = functional_markov_chain(dist, _strategy="refine")
    h_bidir = functional_markov_chain(dist, _strategy="bidirectional")
    h_auto = functional_markov_chain(dist, _strategy="auto")
    assert h_coarsen == pytest.approx(expected)
    assert h_refine == pytest.approx(expected)
    assert h_bidir == pytest.approx(expected)
    assert h_auto == pytest.approx(expected)


def test_fci_dual_search_auto_mdbsi():
    """Auto agrees with coarsen on MDBSI where full refine is impractical."""
    h_coarsen = functional_markov_chain(MDBSI16, _strategy="coarsen")
    h_auto = functional_markov_chain(MDBSI16, _strategy="auto")
    assert h_coarsen == pytest.approx(2.0)
    assert h_auto == pytest.approx(2.0)


def test_meet_immediate_when_g_equals_f():
    """When F = G, the refine path exits without exploring splits."""
    from dit.example_dists import giant_bit

    d = giant_bit(2, 2)
    stats: dict = {}
    h = functional_markov_chain(d, _strategy="refine", _stats=stats)
    assert h == pytest.approx(G(d))
    assert stats["meet_warmstart"] is True
    assert stats["visited"] == 0


def test_auto_router_picks_coarsen_when_f_near_m():
    """Auto routes to coarsen when MSS seed is already optimal."""
    stats: dict = {}
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    functional_markov_chain(d, _strategy="auto", _stats=stats)
    assert stats["route"] == "coarsen"
    assert stats["strategy"] == "coarsen"


def test_auto_router_picks_refine_when_f_equals_g():
    """Auto routes to refine when meet seed is already optimal."""
    from dit.example_dists import giant_bit

    stats: dict = {}
    functional_markov_chain(giant_bit(2, 2), _strategy="auto", _stats=stats)
    assert stats["route"] == "refine"
    assert stats["strategy"] == "refine"


def test_bidirectional_agrees_on_parity():
    """Bidirectional search matches coarsen on parity."""
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    stats: dict = {}
    h = functional_markov_chain(d, _strategy="bidirectional", _stats=stats)
    assert h == pytest.approx(2.0)
    assert stats["direction"] == "bidirectional"


def test_dual_search_visit_counts():
    """Refine visits fewer nodes than coarsen on giant_bit; auto routes sensibly."""
    from dit.example_dists import giant_bit

    stats_coarsen: dict = {}
    stats_refine: dict = {}
    functional_markov_chain(giant_bit(2, 2), _strategy="coarsen", _stats=stats_coarsen)
    functional_markov_chain(giant_bit(2, 2), _strategy="refine", _stats=stats_refine)
    assert stats_refine["visited"] <= stats_coarsen["visited"]

    stats_parity: dict = {}
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    functional_markov_chain(d, _strategy="auto", _stats=stats_parity)
    assert stats_parity["route"] == "coarsen"
    assert stats_parity["strategy"] == "coarsen"
