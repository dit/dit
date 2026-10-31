"""
The conditional-independence union information I_∪^CI of Gomes & Figueiredo (2024).

Each agent i knows p(t) and the channel p(a_i | t) of its source A_i. By sharing
channels (and marginals thereof) the agents can build any distribution

    q(a, t) = p(t) prod_{B in pi} p(a_B | t),

where pi is a partition of the variables in A = A_1 u ... u A_m whose blocks each
lie within some source. Union information is the best such q can do, capped at
the true mutual information:

    I_∪^CI(A_1, ..., A_m -> T) = min{ I(A; T), max_q I_q(A; T) },

and synergy is S^CI(A_1, ..., A_m -> T) = I(A; T) - I_∪^CI(A_1, ..., A_m -> T).

References
----------
.. [1] A. F. C. Gomes and M. A. T. Figueiredo, "A Measure of Synergy Based on
       Union Information", Entropy 26, 271, 2024.
"""

import numpy as np

from ...shannon import conditional_entropy
from ...utils import flatten
from ..pid import BaseSynergyPID

__all__ = (
    "PID_CI",
    "synergy_ci",
    "union_ci",
)


def _prune_sources(d, sources, tol=1e-9):
    """
    Remove sources which are subsets of other sources, then sources which are a
    deterministic function of the variables of the other sources that they do
    not themselves contain.

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    sources : iterable of iterables
        The sources.
    tol : float
        Conditional entropies below `tol` are treated as zero.

    Returns
    -------
    sources : list of tuples
        The pruned sources.
    """
    sources = [tuple(s) for s in sources]
    kept = []
    for i, src in enumerate(sources):
        others = sources[:i] + sources[i + 1 :]
        if any(set(src) < set(o) for o in others) or src in kept:
            continue
        kept.append(src)

    for src in list(kept):
        rest = sorted(set(flatten(s for s in kept if s != src)) - set(src))
        if rest and conditional_entropy(d, list(src), rest) < tol:
            kept.remove(src)

    return kept


def _partitions(variables, sources):
    """
    Enumerate set partitions of `variables` in which every block lies within a source.

    Parameters
    ----------
    variables : tuple
        The variables to partition.
    sources : list of tuples
        The sources.

    Yields
    ------
    partition : list of tuples
        A partition of `variables`.
    """
    if not variables:
        yield []
        return
    first, rest = variables[0], variables[1:]
    blocks = set()
    for src in sources:
        if first in src:
            pool = [v for v in rest if v in src]
            for mask in range(2 ** len(pool)):
                blocks.add((first,) + tuple(v for j, v in enumerate(pool) if mask >> j & 1))
    for block in sorted(blocks):
        remaining = tuple(v for v in rest if v not in block)
        for tail in _partitions(remaining, sources):
            yield [block, *tail]


def _joint_array(d, variables, target):
    """
    The joint pmf of `variables` (one axis each) and `target` (last axis) as a dense array.
    """
    d = d.coalesce([[v] for v in variables] + [list(target)])
    alphabets = [sorted({o[i] for o in d.outcomes}) for i in range(len(variables) + 1)]
    index = [{x: j for j, x in enumerate(alpha)} for alpha in alphabets]
    arr = np.zeros([len(alpha) for alpha in alphabets])
    for outcome, p in d.zipped():
        arr[tuple(index[i][x] for i, x in enumerate(outcome))] += p
    return arr


def _mutual_information(arr):
    """
    I(A; T) in bits, where T is the last axis of `arr` and A is all the others.
    """
    joint = arr.reshape(-1, arr.shape[-1])
    p_a = np.broadcast_to(joint.sum(axis=1, keepdims=True), joint.shape)
    p_t = np.broadcast_to(joint.sum(axis=0, keepdims=True), joint.shape)
    mask = joint > 0
    # A difference of logarithms, since p(a) p(t) underflows when both are tiny.
    terms = np.log2(joint[mask]) - np.log2(p_a[mask]) - np.log2(p_t[mask])
    return float(np.sum(joint[mask] * terms))


def _product_distribution(arr, partition):
    """
    Build q(a, t) = p(t) prod_B p(a_B | t) from the joint array `arr`.
    """
    n = arr.ndim - 1
    p_t = arr.sum(axis=tuple(range(n)), keepdims=True)
    safe = np.where(p_t > 0, p_t, 1)
    q = p_t.copy()
    for block in partition:
        others = tuple(i for i in range(n) if i not in block)
        q = q * (arr.sum(axis=others, keepdims=True) / safe)
    return q


def union_ci(d, sources, target):
    """
    Compute the conditional-independence union information I_∪^CI.

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    union : float
        I_∪^CI(sources -> target), in bits.

    Notes
    -----
    Following Definition 1 of :cite:`gomes2024measure`, sources contained in
    another source are dropped, as are sources which are a deterministic function
    of the other sources. For overlapping sources, a source is only dropped if it
    is determined by the variables of the other sources that it does not contain.
    """
    sources = _prune_sources(d, sources)
    variables = tuple(sorted(set(flatten(sources))))
    if not variables:
        return 0.0
    arr = _joint_array(d, variables, target)
    sources = [tuple(variables.index(v) for v in src) for src in sources]
    whole = _mutual_information(arr)
    best = max(
        _mutual_information(_product_distribution(arr, part))
        for part in _partitions(tuple(range(len(variables))), sources)
    )
    return min(whole, best)


def synergy_ci(d, sources, target):
    """
    Compute the conditional-independence synergy S^CI = I(A; T) - I_∪^CI.

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    syn : float
        S^CI(sources -> target), in bits.
    """
    variables = sorted(set(flatten(sources)))
    whole = _mutual_information(_joint_array(d, variables, target))
    return whole - union_ci(d, sources, target)


class PID_CI(BaseSynergyPID):
    """
    The conditional-independence union information I_∪^CI of Gomes & Figueiredo (2024).

    Synergy is the information lost by assuming the sources are conditionally
    independent given the target; redundancies follow by inclusion-exclusion
    over union informations.

    References
    ----------
    A. F. C. Gomes and M. A. T. Figueiredo, "A Measure of Synergy Based on
    Union Information", Entropy 26, 271, 2024.
    """

    _name = "I_∪^CI"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute S^CI for a collection of sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        syn : float
            S^CI(sources -> target), in bits.
        """
        return synergy_ci(d, sources, target)
