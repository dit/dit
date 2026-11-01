"""
Tests for dit.algorithms.caekl_psp.
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms.caekl_psp import (
    caekl_from_partition_pmf,
    caekl_mutual_information_psp_pmf,
    caekl_mutual_information_psp_pmf_grad_data,
    caekl_mutual_information_psp_pmf_with_partition,
    caekl_partition_indices,
    labels_from_partition,
    partition_from_labels,
)
from dit.multivariate import caekl_mutual_information, entropy, mmi_psp
from dit.utils import partitions


def _h(pmf):
    """Shannon entropy, in bits, of an array of probabilities."""
    p = np.asarray(pmf, dtype=float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _sum_axes(pmf, axes):
    return np.asarray(pmf).sum(axis=axes, keepdims=True)


def _binary_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _xor_pmf():
    """X0, X1 uniform bits, X2 = X0 xor X1 and X3 = X0, as a (2, 2, 2, 2) array."""
    pmf = np.zeros((2, 2, 2, 2))
    for x0, x1 in product(range(2), repeat=2):
        pmf[x0, x1, x0 ^ x1, x0] = 0.25
    return pmf


def _noisy_copy_pmf():
    """X0 = X1 = A uniform, X2 = A xor N, X3 = N xor M with N ~ Bern(0.1), M ~ Bern(0.2)."""
    pmf = np.zeros((2, 2, 2, 2))
    for a, n, m in product(range(2), repeat=3):
        pmf[a, a, a ^ n, n ^ m] += 0.5 * (0.1 if n else 0.9) * (0.2 if m else 0.8)
    return pmf


def _random_pmf():
    return np.random.default_rng(0).dirichlet(np.ones(24)).reshape(2, 3, 2, 2)


def _kept_axes(pmf, axes):
    """The axes of ``pmf`` left after summing out ``axes`` (every axis has size > 1)."""
    return {i for i, n in enumerate(_sum_axes(pmf, axes).shape) if n > 1}


# (pmf, rvs, crvs, closed-form CAEKL or None, optimal partition or None)
CASES = {
    "xor": (_xor_pmf().sum(axis=3), {0, 1, 2}, set(), 0.5, {frozenset({0}), frozenset({1}), frozenset({2})}),
    "xor given copy": (_xor_pmf(), {0, 1, 2}, {3}, 0.0, {frozenset({0}), frozenset({1, 2})}),
    "noisy copy": (_noisy_copy_pmf(), {0, 1, 2}, set(), 1 - _binary_entropy(0.1), {frozenset({0, 1}), frozenset({2})}),
    "noisy copy given noise": (
        _noisy_copy_pmf(),
        {0, 1, 2},
        {3},
        1 - _binary_entropy(0.1) - _binary_entropy(0.2) + _binary_entropy(0.26),
        {frozenset({0, 1}), frozenset({2})},
    ),
    "random, crvs in the middle": (_random_pmf(), {0, 1, 3}, {2}, None, None),
    "random, marginalized": (_random_pmf(), {0, 2}, set(), None, None),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_caekl_from_selected_partition(case):
    """
    The PSP partition and its marginalization indices give the CAEKL mutual information of
    dit.multivariate, the closed form, and the minimum of I_P over all partitions.
    """
    pmf, rvs, crvs, closed_form, optimal = CASES[case]
    all_vars = set(range(pmf.ndim))
    kwargs = {"all_vars": all_vars, "rvs": rvs, "crvs": crvs, "h": _h, "sum_axes": _sum_axes}

    partition, idx_joint, idx_crvs, idx_parts = caekl_partition_indices(pmf, **kwargs)
    value = caekl_from_partition_pmf(
        pmf, partition, idx_joint=idx_joint, idx_crvs=idx_crvs, idx_parts=idx_parts, h=_h, sum_axes=_sum_axes
    )

    dist = Distribution.from_ndarray(pmf)
    expected = caekl_mutual_information(dist, [[i] for i in sorted(rvs)], sorted(crvs))
    assert value == pytest.approx(expected, abs=1e-10)
    if closed_form is not None:
        assert value == pytest.approx(closed_form, abs=1e-10)
    if optimal is not None:
        assert set(partition) == optimal

    # the blocks partition rvs, and each index tuple sums out everything but its marginal
    assert len(partition) >= 2
    assert frozenset().union(*partition) == rvs
    assert sum(len(block) for block in partition) == len(rvs)
    assert _kept_axes(pmf, idx_joint) == rvs | crvs
    assert _kept_axes(pmf, idx_crvs) == crvs
    assert set(idx_parts) == set(partition)
    for block in partition:
        assert _kept_axes(pmf, idx_parts[block]) == set(block) | crvs

    # brute force: min over partitions of (sum_B H(X_B | Z) - H(X | Z)) / (|P| - 1)
    h_crvs = _h(pmf.sum(axis=tuple(all_vars - crvs)))

    def cond_h(variables):
        return _h(pmf.sum(axis=tuple(all_vars - (set(variables) | crvs)))) - h_crvs

    brute = min(
        (sum(cond_h(block) for block in part) - cond_h(rvs)) / (len(part) - 1)
        for part in partitions(sorted(rvs))
        if len(part) > 1
    )
    assert value == pytest.approx(brute, abs=1e-10)

    # the other entry points select the same partition and value
    grad_data = caekl_mutual_information_psp_pmf_grad_data(pmf, **kwargs)
    assert grad_data[1:] == (partition, idx_joint, idx_crvs, idx_parts)
    assert grad_data[0] == pytest.approx(value, abs=1e-12)
    assert caekl_mutual_information_psp_pmf(pmf, **kwargs) == pytest.approx(value, abs=1e-12)


def test_partition_information_of_every_partition():
    """
    For any fixed partition caekl_from_partition_pmf is I_P, and the PSP partition is a minimizer.
    """
    pmf = _noisy_copy_pmf()
    rvs, crvs = {0, 1, 2}, {3}
    all_vars = set(range(pmf.ndim))
    best, idx_joint, idx_crvs, _ = caekl_partition_indices(
        pmf, all_vars=all_vars, rvs=rvs, crvs=crvs, h=_h, sum_axes=_sum_axes
    )
    dist = Distribution.from_ndarray(pmf)
    h_all = entropy(dist, [0, 1, 2], [3])

    values = {}
    for part in partitions(sorted(rvs)):
        if len(part) < 2:
            continue
        part = tuple(sorted(part, key=min))
        idx_parts = {block: tuple(all_vars - (block | crvs)) for block in part}
        value = caekl_from_partition_pmf(
            pmf, part, idx_joint=idx_joint, idx_crvs=idx_crvs, idx_parts=idx_parts, h=_h, sum_axes=_sum_axes
        )
        expected = (sum(entropy(dist, sorted(block), [3]) for block in part) - h_all) / (len(part) - 1)
        assert value == pytest.approx(expected, abs=1e-10)
        values[frozenset(part)] = value

    assert len(values) == 4
    assert values[frozenset(best)] == pytest.approx(min(values.values()), abs=1e-12)
    # the noisy copy is far from the other partitions
    assert sorted(values.values())[1] - values[frozenset(best)] > 0.1


def test_group_entropy_oracle(monkeypatch):
    """
    PSP is handed a normalized oracle: h(empty set) = 0 and h(S) = H(X_S | crvs) for unions
    of the (sorted) random variables.
    """
    original = mmi_psp.caekl_mutual_information_psp_with_partition
    seen = {}

    def spy(h, n_groups):
        seen["h"], seen["n_groups"] = h, n_groups
        return original(h, n_groups)

    monkeypatch.setattr(mmi_psp, "caekl_mutual_information_psp_with_partition", spy)
    pmf = _random_pmf()
    rvs, crvs = (0, 1, 3), {2}
    value, partition = caekl_mutual_information_psp_pmf_with_partition(
        pmf, all_vars={0, 1, 2, 3}, rvs=set(rvs), crvs=crvs, h=_h, sum_axes=_sum_axes
    )

    h = seen["h"]
    assert seen["n_groups"] == 3
    assert h(frozenset()) == 0.0
    dist = Distribution.from_ndarray(pmf)
    for groups in product([False, True], repeat=3):
        subset = frozenset(i for i, keep in enumerate(groups) if keep)
        if subset:
            variables = [rvs[i] for i in sorted(subset)]
            assert h(subset) == pytest.approx(entropy(dist, variables, [2]), abs=1e-10)
    assert value == pytest.approx(caekl_mutual_information(dist, [[0], [1], [3]], [2]), abs=1e-10)
    assert frozenset().union(*partition) == set(rvs)


def test_labels_from_partition():
    """Labels give the block of each random variable, in the order of rvs_sorted."""
    rvs_sorted = (0, 2, 5)
    partition = (frozenset({0, 5}), frozenset({2}))
    labels = labels_from_partition(partition, rvs_sorted)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, [0, 1, 0])
    assert partition_from_labels(labels, rvs_sorted) == partition


def test_partition_labels_round_trip():
    """Every partition survives encoding to labels and back, block order included."""
    rvs_sorted = (1, 3, 4, 7)
    parts = list(partitions(rvs_sorted))
    assert len(parts) == 15  # Bell number B_4
    for part in parts:
        part = tuple(sorted(part, key=min))
        labels = labels_from_partition(part, rvs_sorted)
        assert labels.shape == (4,)
        assert partition_from_labels(labels, rvs_sorted) == part


def test_partition_from_labels_relabels():
    """
    Arbitrary labels decode to the blocks of equal labels, ordered by label; re-encoding
    yields the canonical labels 0, 1, ... and the same partition.
    """
    rvs_sorted = (1, 3, 4, 6, 8)
    partition = partition_from_labels([7, 2, 7, 2, 9], rvs_sorted)
    assert partition == (frozenset({3, 6}), frozenset({1, 4}), frozenset({8}))
    np.testing.assert_array_equal(labels_from_partition(partition, rvs_sorted), [1, 0, 1, 0, 2])

    for labels in product(range(3), repeat=4):
        partition = partition_from_labels(np.array(labels), rvs_sorted[:4])
        block_of = {rv: i for i, block in enumerate(partition) for rv in block}
        assert sorted(block_of) == list(rvs_sorted[:4])
        for i, j in product(range(4), repeat=2):
            same_block = block_of[rvs_sorted[i]] == block_of[rvs_sorted[j]]
            assert same_block == (labels[i] == labels[j])
        canonical = labels_from_partition(partition, rvs_sorted[:4])
        assert partition_from_labels(canonical, rvs_sorted[:4]) == partition
