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
from dit.multivariate.common_informations._functional_partition import (
    conditional_dtc,
    labels_from_partition,
    partition_entropy,
    prepare_functional_search,
)
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
        frozenset({frozenset([("0", "0", "0")]), frozenset([("0", "1", "1")]), frozenset([("1", "0", "1")]), frozenset([("1", "1", "0")])}),
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
