"""
The functional common information.
"""

import heapq
from itertools import combinations

import numpy as np

from ...distconst import RVFunctions, insert_rvf, modify_outcomes
from ...helpers import normalize_rvs
from ...utils import partitions, unitful
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy
from ._functional_partition import (
    conditional_dtc,
    labels_from_partition,
    partition_entropy,
    prepare_functional_search,
)

__all__ = ("functional_common_information",)


def functional_markov_chain_naive(dist, rvs=None, crvs=None):  # pragma: no cover
    """
    Add the smallest function of `dist` which renders `rvs` independent.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the smallest function will be constructed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    d : Distribution
        The distribution `dist` with the additional variable added to the end.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    outcomes = dist.outcomes
    bf = RVFunctions(dist)
    f = [len(dist.rvs)]
    parts = partitions(outcomes)
    dists = [insert_rvf(dist, bf.from_partition(part)) for part in parts]
    B = lambda d: dual_total_correlation(d, rvs, crvs + f)
    dists = [d for d in dists if np.isclose(B(d), 0)]
    return min(dists, key=lambda d: entropy(d, rvs=f))


def functional_markov_chain(dist, rvs=None, crvs=None):
    """
    Return H(W) for the smallest function W of `dist` which renders `rvs` independent.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the smallest function will be constructed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    h : float
        The entropy of the smallest valid functional Markov variable W.

    Notes
    -----
    The search explores coarsenings of the finest outcome partition among those
    with zero dual total correlation B(rvs | crvs, W).  Per-partition evaluation
    uses numpy PMF marginals (O(|support| · n_rvs)) rather than rebuilding a
    :class:`~dit.Distribution`.  Partitions are processed in best-first order by
    H(W) so the loop can stop as soon as H(W) equals B(rvs | crvs).

    The number of valid partitions can still grow quickly with support size; it
    remains an open question whether a direct construction exists (as for the GK
    or MSS common variables).  See james2017multivariate.
    """
    optimal_b = dual_total_correlation(dist, rvs, crvs)
    ctx = prepare_functional_search(dist, rvs=rvs, crvs=crvs)
    pmf_size = int(np.prod(ctx.shape))

    part = frozenset(frozenset([o]) for o in ctx.dist.outcomes)
    finest_labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)
    optimal_h = partition_entropy(ctx.pmf, finest_labels)

    heap: list[tuple[float, frozenset]] = [(optimal_h, part)]
    checked: set[frozenset] = set()

    while heap:  # pragma: no branch
        _, part = heapq.heappop(heap)

        if part in checked:
            continue
        checked.add(part)

        labels = labels_from_partition(part, ctx.outcome_to_flat, pmf_size)

        if not np.isclose(conditional_dtc(ctx.pmf, labels, ctx.rvs, ctx.crvs), 0):
            continue

        h = partition_entropy(ctx.pmf, labels)

        if h <= optimal_h:
            optimal_h = h

        if np.isclose(h, optimal_b):
            break

        new_parts = [
            frozenset([p for p in part if p not in pair] + [pair[0] | pair[1]]) for pair in combinations(part, 2)
        ]
        for new_part in new_parts:
            if new_part in checked:
                continue
            new_labels = labels_from_partition(new_part, ctx.outcome_to_flat, pmf_size)
            new_h = partition_entropy(ctx.pmf, new_labels)
            heapq.heappush(heap, (new_h, new_part))

    return optimal_h


@unitful
def functional_common_information(dist, rvs=None, crvs=None):
    """
    Compute the functional common information, F, of `dist`. It is the entropy
    of the smallest random variable W such that all the variables in `rvs` are
    rendered independent conditioned on W, and W is a function of `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the functional common information is
        computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    F : float
        The functional common information.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    dtc = dual_total_correlation(dist, rvs, crvs)
    ent = entropy(dist, rvs, crvs)
    if np.isclose(dtc, ent):
        return dtc

    return functional_markov_chain(dist, rvs, crvs)
