"""
CAEKL via PSP on marginalized PMF arrays (optimization backends).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache

import numpy as np

EntropyFn = Callable[..., float]


def _group_rvs_from_sets(rvs: Iterable[int]) -> list[frozenset[int]]:
    return [frozenset({rv}) for rv in sorted(rvs)]


def _cluster_to_rv_part(
    partition: tuple[frozenset[int], ...],
    group_rvs: list[frozenset[int]],
) -> tuple[frozenset[int], ...]:
    blocks: list[frozenset[int]] = []
    for cluster in partition:
        block: set[int] = set()
        for group_idx in cluster:
            block |= set(group_rvs[group_idx])
        blocks.append(frozenset(block))
    return tuple(blocks)


def caekl_mutual_information_psp_pmf(
    pmf,
    *,
    all_vars: set[int],
    rvs: set[int],
    crvs: set[int],
    h: EntropyFn,
    sum_axes,
) -> float:
    """
    CAEKL of ``rvs`` given ``crvs`` for a joint ``pmf`` using PSP.

    Parameters
    ----------
    pmf : array
        Full joint PMF including auxiliary variables.
    all_vars : set[int]
        All variable indices in ``pmf``.
    rvs : set[int]
        Target random-variable indices (one group per index).
    crvs : set[int]
        Conditioning variable indices.
    h : callable
        Shannon entropy ``h(marginal_pmf)``.
    sum_axes : callable
        ``sum_axes(pmf, axes)`` marginalizes ``pmf`` over ``axes``.
    """
    value, _ = caekl_mutual_information_psp_pmf_with_partition(
        pmf,
        all_vars=all_vars,
        rvs=rvs,
        crvs=crvs,
        h=h,
        sum_axes=sum_axes,
    )
    return value


def caekl_mutual_information_psp_pmf_with_partition(
    pmf,
    *,
    all_vars: set[int],
    rvs: set[int],
    crvs: set[int],
    h: EntropyFn,
    sum_axes,
) -> tuple[float, tuple[frozenset[int], ...]]:
    """
    CAEKL via PSP on a PMF, returning an optimal partition over ``rvs``.

    The partition is expressed as a tuple of frozensets of elements of ``rvs``.
    """
    group_rvs = _group_rvs_from_sets(rvs)
    n_groups = len(group_rvs)
    idx_crvs = tuple(all_vars - crvs)
    pmf_crvs = sum_axes(pmf, idx_crvs)

    def conditional_h(vars_set: set[int]) -> float:
        idx = tuple(all_vars - (vars_set | crvs))
        pmf_joint = sum_axes(pmf, idx)
        return h(pmf_joint) - h(pmf_crvs)

    @lru_cache(maxsize=None)
    def h_groups(group_indices: frozenset[int]) -> float:
        if not group_indices:
            return 0.0
        vars_set: set[int] = set()
        for group_idx in group_indices:
            vars_set |= set(group_rvs[group_idx])
        return conditional_h(vars_set)

    from ..multivariate.mmi_psp import caekl_mutual_information_psp_with_partition

    value, partition = caekl_mutual_information_psp_with_partition(h_groups, n_groups)
    return value, _cluster_to_rv_part(partition, group_rvs)


def caekl_mutual_information_psp_pmf_grad_data(
    pmf,
    *,
    all_vars: set[int],
    rvs: set[int],
    crvs: set[int],
    h: EntropyFn,
    sum_axes,
) -> tuple[float, tuple[frozenset[int], ...], tuple[int, ...], tuple[int, ...], dict[frozenset[int], tuple[int, ...]]]:
    """
    CAEKL value, optimal partition, and marginalization indices for gradients.

    Returns
    -------
    value : float
    partition : tuple of frozenset[int]
        Optimal partition blocks (subsets of ``rvs``).
    idx_joint : tuple[int, ...]
    idx_crvs : tuple[int, ...]
    idx_parts : dict
        Maps each block in ``partition`` to axes to sum for its marginal PMF.
    """
    value, partition = caekl_mutual_information_psp_pmf_with_partition(
        pmf,
        all_vars=all_vars,
        rvs=rvs,
        crvs=crvs,
        h=h,
        sum_axes=sum_axes,
    )
    idx_joint = tuple(all_vars - (rvs | crvs))
    idx_crvs = tuple(all_vars - crvs)
    idx_parts = {block: tuple(all_vars - (set(block) | crvs)) for block in partition}
    return value, partition, idx_joint, idx_crvs, idx_parts


def labels_from_partition(
    partition: tuple[frozenset[int], ...],
    rvs_sorted: tuple[int, ...],
) -> np.ndarray:
    """Encode a CAEKL partition as per-RV block labels (for JAX host callbacks)."""
    labels = np.zeros(len(rvs_sorted), dtype=np.int32)
    rv_index = {rv: i for i, rv in enumerate(rvs_sorted)}
    for block_id, block in enumerate(partition):
        for rv in block:
            labels[rv_index[rv]] = block_id
    return labels


def partition_from_labels(
    labels: np.ndarray,
    rvs_sorted: tuple[int, ...],
) -> tuple[frozenset[int], ...]:
    """Decode block labels into a CAEKL partition over ``rvs_sorted``."""
    labels = np.asarray(labels, dtype=np.int32)
    blocks: dict[int, set[int]] = {}
    for i, lab in enumerate(labels):
        blocks.setdefault(int(lab), set()).add(rvs_sorted[i])
    return tuple(frozenset(blocks[k]) for k in sorted(blocks))


def caekl_partition_indices(
    pmf,
    *,
    all_vars: set[int],
    rvs: set[int],
    crvs: set[int],
    h: EntropyFn,
    sum_axes,
) -> tuple[tuple[frozenset[int], ...], tuple[int, ...], tuple[int, ...], dict[frozenset[int], tuple[int, ...]]]:
    """
    PSP optimal partition and marginalization indices for autodiff backends.

    Returns ``(partition, idx_joint, idx_crvs, idx_parts)``.
    """
    _, partition = caekl_mutual_information_psp_pmf_with_partition(
        pmf,
        all_vars=all_vars,
        rvs=rvs,
        crvs=crvs,
        h=h,
        sum_axes=sum_axes,
    )
    idx_joint = tuple(all_vars - (rvs | crvs))
    idx_crvs = tuple(all_vars - crvs)
    idx_parts = {block: tuple(all_vars - (set(block) | crvs)) for block in partition}
    return partition, idx_joint, idx_crvs, idx_parts


def caekl_from_partition_pmf(
    pmf,
    partition: tuple[frozenset[int], ...],
    *,
    idx_joint: tuple[int, ...],
    idx_crvs: tuple[int, ...],
    idx_parts: dict[frozenset[int], tuple[int, ...]],
    h: EntropyFn,
    sum_axes,
) -> float:
    """Partition information ``I_P`` for a fixed partition (used after PSP selects ``P``)."""
    norm = len(partition) - 1
    pmf_joint = sum_axes(pmf, idx_joint)
    pmf_crvs = sum_axes(pmf, idx_crvs)
    h_crvs = h(pmf_crvs)
    h_joint = h(pmf_joint) - h_crvs
    h_parts = sum(h(sum_axes(pmf, idx_parts[block])) - h_crvs for block in partition)
    return (h_parts - h_joint) / norm
