"""
Partition-label helpers for functional common information search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...distconst import modify_outcomes
from ...helpers import normalize_rvs, parse_rvs
from ...shannon import entropy_pmf

__all__ = (
    "FunctionalSearchContext",
    "conditional_dtc",
    "labels_from_partition",
    "partition_entropy",
    "prepare_functional_search",
)


def _entropy_array(pmf: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.nansum(pmf * np.log2(pmf)))


def partition_entropy(pmf: np.ndarray, labels: np.ndarray) -> float:
    """
    Shannon entropy of the block weights induced by ``labels`` on ``pmf``.
    """
    flat = pmf.ravel()
    n_blocks = int(labels.max()) + 1 if labels.size else 0
    if n_blocks == 0:
        return 0.0
    block_probs = np.bincount(labels, weights=flat, minlength=n_blocks)
    return float(entropy_pmf(block_probs))


def _build_pmf_aug(pmf: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shape = pmf.shape
    n_blocks = int(labels.max()) + 1
    pmf_aug = np.zeros(shape + (n_blocks,))

    flat_pmf = pmf.ravel()
    nz = flat_pmf > 0
    if not np.any(nz):
        return pmf_aug

    flat_indices = np.flatnonzero(nz)
    multi = np.array(np.unravel_index(flat_indices, shape))
    w = labels[flat_indices]
    pmf_aug[tuple(multi.tolist()) + (w,)] = flat_pmf[nz]
    return pmf_aug


def conditional_dtc(
    pmf: np.ndarray,
    labels: np.ndarray,
    rvs: list[list[int]],
    crvs: list[int],
) -> float:
    """
    Dual total correlation of ``rvs`` given ``crvs`` and the partition ``labels``.

    ``labels`` defines a deterministic auxiliary W = f(joint outcome) on the
    last axis of an augmented PMF.
    """
    pmf_aug = _build_pmf_aug(pmf, labels)
    w_axis = len(pmf.shape)
    all_vars = set(range(w_axis + 1))
    rv_sets = [set(rv) for rv in rvs]
    crvs_ext = set(crvs) | {w_axis}
    all_rvs = set().union(*rv_sets)

    def cond_h(var_set: set[int], cond_set: set[int]) -> float:
        idx_joint = tuple(all_vars - (var_set | cond_set))
        idx_crvs = tuple(all_vars - cond_set)
        pmf_joint = pmf_aug.sum(axis=idx_joint, keepdims=True)
        pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)
        return _entropy_array(pmf_joint) - _entropy_array(pmf_crvs)

    one = cond_h(all_rvs, crvs_ext)
    two = sum(cond_h(rv, (all_rvs - rv) | crvs_ext) for rv in rv_sets)
    return one - two


@dataclass(frozen=True)
class FunctionalSearchContext:
    """Preprocessed joint used throughout a functional-Markov partition search."""

    dist: object
    pmf: np.ndarray
    shape: tuple[int, ...]
    outcome_to_flat: dict[object, int]
    rvs: list[list[int]]
    crvs: list[int]


def prepare_functional_search(dist, rvs=None, crvs=None) -> FunctionalSearchContext:
    """
    Copy, sparsify, and densify ``dist``; return PMF arrays and outcome maps.
    """
    rv_names = dist.get_rv_names()
    dist = dist.copy()
    dist.make_sparse()
    dist = modify_outcomes(dist, tuple)
    if rv_names is not None:
        dist.set_rv_names(rv_names)

    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    rvs = [list(parse_rvs(dist, rv)[1]) for rv in rvs]
    crvs = list(parse_rvs(dist, crvs)[1])

    dist.make_dense()
    shape = tuple(len(a) for a in dist.alphabet)
    pmf = dist.pmf.reshape(shape)

    alphabets = dist.alphabet
    outcome_to_flat: dict[object, int] = {}
    for outcome in dist.outcomes:
        idx = tuple(alphabets[i].index(outcome[i]) for i in range(len(outcome)))
        outcome_to_flat[outcome] = int(np.ravel_multi_index(idx, shape))

    return FunctionalSearchContext(
        dist=dist,
        pmf=pmf,
        shape=shape,
        outcome_to_flat=outcome_to_flat,
        rvs=rvs,
        crvs=crvs,
    )


def labels_from_partition(
    partition: frozenset[frozenset],
    outcome_to_flat: dict[object, int],
    size: int,
) -> np.ndarray:
    """Encode a partition of outcomes as per-flat-index block labels."""
    labels = np.zeros(size, dtype=np.int32)
    for block_id, block in enumerate(partition):
        for outcome in block:
            labels[outcome_to_flat[outcome]] = block_id
    return labels
