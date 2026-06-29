"""
Polynomial-time CAEKL / multivariate mutual information via agglomerative PSP.

Implements the Fuse subroutine and agglomerative info-clustering loop from
Chan and Liu :cite:`ChanLiu2017agglomerative`, using minimum-norm-base
subroutines on entropy-derived submodular functions.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from ..algorithms.submodular import minimum_norm_base
from ..helpers import normalize_rvs
from .entropy import entropy

__all__ = (
    "caekl_mutual_information_psp",
    "caekl_mutual_information_psp_with_partition",
)

GroupEntropy = Callable[[frozenset[int]], float]


class _EntropyOracle:
    """Cached conditional entropy on unions of indexed groups."""

    def __init__(self, h: GroupEntropy):
        self._h = h

    @lru_cache(maxsize=None)
    def h(self, group_indices: frozenset[int]) -> float:
        return self._h(group_indices)


def _h_on_partition(partition: tuple[frozenset[int], ...], cluster_indices: frozenset[int], oracle: _EntropyOracle) -> float:
    rvs_groups: set[int] = set()
    for idx in cluster_indices:
        rvs_groups |= partition[idx]
    return oracle.h(frozenset(rvs_groups))


def _make_g_j(
    partition: tuple[frozenset[int], ...],
    j: int,
    oracle: _EntropyOracle,
):
    """Normalized submodular ``g_j`` on ``{j+1, ..., n-1}`` for the current partition."""

    def g_j(subset: frozenset[int]) -> float:
        if not subset.issubset(frozenset(range(j + 1, len(partition)))):
            msg = f"g_j subset {subset} not contained in tail of {j}"
            raise ValueError(msg)
        union = subset | {j}
        return _h_on_partition(partition, frozenset(union), oracle) - sum(
            _h_on_partition(partition, frozenset({i}), oracle) for i in union
        )

    return g_j


def _partition_information(
    partition: tuple[frozenset[int], ...],
    *,
    oracle: _EntropyOracle,
    n_groups: int | None = None,
) -> float:
    """Partition information ``I_P`` for a cluster partition of groups."""
    norm = len(partition) - 1
    if norm <= 0:
        msg = "partition information requires at least two blocks"
        raise ValueError(msg)
    if n_groups is None:
        all_groups = frozenset().union(*partition)
    else:
        all_groups = frozenset(range(n_groups))
    h_all = oracle.h(all_groups)
    h_parts = sum(oracle.h(block) for block in partition)
    return (h_parts - h_all) / norm


def _fuse_merge_partition(
    partition: list[frozenset[int]],
    merge: set[int],
) -> list[frozenset[int]]:
    merged_groups: set[int] = set()
    new_partition: list[frozenset[int]] = []
    for idx, cluster in enumerate(partition):
        if idx in merge:
            merged_groups |= cluster
        else:
            new_partition.append(cluster)
    new_partition.append(frozenset(merged_groups))
    return new_partition


def fuse(partition: list[frozenset[int]], oracle: _EntropyOracle) -> tuple[float, list[frozenset[int]]]:
    """
    One agglomerative Fuse step (Algorithm 2, Chan--Liu 2017).

    Parameters
    ----------
    partition : list of frozenset[int]
        Current partition; each frozenset lists group indices in that cluster.
    oracle : _EntropyOracle
        Entropy oracle for the underlying distribution.

    Returns
    -------
    gamma : float
        Critical threshold ``I*`` for this agglomeration level.
    new_partition : list
        Coarser partition after merging ``C*``.

    Notes
    -----
    When several ``j`` tie for ``min_i x^{(j)}_i``, the candidate whose merge
    minimizes the immediate partition information ``I_P`` is chosen (then the
    smallest ``j``).  This matches Chan--Liu :cite:`ChanLiu2017agglomerative`
    on exact ties and avoids bad merges when entropies are numerically zero.
    """
    n = len(partition)
    if n <= 1:
        return 0.0, partition

    partition_tuple = tuple(partition)
    bases: dict[int, dict[int, float]] = {}

    for j in range(n):
        ground = tuple(range(j + 1, n))
        if not ground:
            continue
        g_j = _make_g_j(partition_tuple, j, oracle)
        bases[j] = minimum_norm_base(g_j, ground)

    best_j = None
    best_min = float("inf")
    for j, base in bases.items():
        local_min = min(base.values())
        if local_min < best_min:
            best_min = local_min

    if best_min == float("inf"):
        return 0.0, partition

    gamma = -best_min
    tol = 1e-10

    candidates: list[tuple[float, int, list[frozenset[int]]]] = []
    for j, base in bases.items():
        local_min = min(base.values())
        if local_min > best_min + tol:
            continue
        min_x = local_min
        merge = {j}
        merge.update(i for i, val in base.items() if val <= min_x + tol)
        new_partition = _fuse_merge_partition(partition, merge)
        score = (
            _partition_information(tuple(new_partition), oracle=oracle)
            if len(new_partition) > 1
            else float("inf")
        )
        candidates.append((score, j, new_partition))

    _, _, new_partition = min(candidates, key=lambda item: (item[0], item[1]))

    return gamma, new_partition


def caekl_mutual_information_psp_with_partition(
    h: GroupEntropy,
    n_groups: int,
) -> tuple[float, tuple[frozenset[int], ...]]:
    """
    CAEKL / MMI via agglomerative PSP, also returning the optimal partition.

    The agglomerative Fuse chain visits a principal sequence of partitions from
    all singletons toward the trivial one-block partition.  CAEKL is
    ``min_P I_P``; along this chain the minimum is attained at one of the
    visited partitions (including the initial singleton partition), so we track
    the best ``I_P`` seen rather than returning only the last Fuse threshold.

    Parameters
    ----------
    h : callable
        ``h(S)`` returns conditional entropy ``H(Z_S | crvs)`` for a set ``S`` of
        group indices ``{0, ..., n_groups - 1}``.
    n_groups : int
        Number of variable groups.

    Returns
    -------
    value : float
        CAEKL mutual information.
    partition : tuple of frozenset[int]
        An optimal partition of the groups (each frozenset lists group indices in
        one block). Valid subgradient partition for the pmf optimizer.
    """
    if n_groups <= 1:
        msg = "CAEKL requires at least two random-variable groups"
        raise ValueError(msg)

    oracle = _EntropyOracle(h)
    partition = [frozenset({i}) for i in range(n_groups)]
    partition_tuple = tuple(partition)
    best_value = _partition_information(partition_tuple, oracle=oracle, n_groups=n_groups)
    best_partition = partition_tuple

    while len(partition) > 1:
        _, partition = fuse(partition, oracle)
        if len(partition) > 1:
            partition_tuple = tuple(partition)
            value = _partition_information(partition_tuple, oracle=oracle, n_groups=n_groups)
            if value < best_value:
                best_value = value
                best_partition = partition_tuple

    return best_value, best_partition


def caekl_mutual_information_psp(dist, rvs, crvs):
    """
    CAEKL mutual information via agglomerative PSP (Chan--Liu Fuse).

    Parameters
    ----------
    dist : Distribution
        Joint distribution.
    rvs : list
        Random-variable groups (same convention as :func:`caekl_mutual_information`).
    crvs : list
        Conditioning variables.

    Returns
    -------
    J : float
        The CAEKL mutual information.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    n = len(rvs)

    @lru_cache(maxsize=None)
    def h(group_indices: frozenset[int]) -> float:
        if not group_indices:
            return 0.0
        variables: set[int] = set()
        for idx in group_indices:
            variables.update(rvs[idx])
        return entropy(dist, [list(variables)], crvs)

    value, _ = caekl_mutual_information_psp_with_partition(h, n)
    return value
