"""
The functional common information.
"""

from collections import deque
from itertools import combinations

import numpy as np

from ...distconst import RVFunctions, insert_rvf, modify_outcomes
from ...helpers import normalize_rvs, parse_rvs
from ...utils import partitions, unitful
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy

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

    Notes
    -----
    The implementation of this function is quite slow. It is approximately
    doubly exponential in the size of the sample space. This method is several
    times faster than the naive method however. It remains an open question as
    to whether a method to directly construct this variable exists (as it does
    with the GK common variable, minimal sufficient statistic, etc).
    """
    optimal_b = dual_total_correlation(dist, rvs, crvs)

    rv_names = dist.get_rv_names()
    dist = modify_outcomes(dist, tuple)
    if rv_names is not None:
        dist.set_rv_names(rv_names)

    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    rvs = [parse_rvs(dist, rv)[1] for rv in rvs]
    crvs = parse_rvs(dist, crvs)[1]

    part = frozenset(frozenset([o]) for o in dist.outcomes)  # make copy

    bf = RVFunctions(dist)

    W = (dist.outcome_length(),)

    H = lambda d: entropy(d, W)
    B = lambda d: dual_total_correlation(d, rvs, crvs + W)

    initial = insert_rvf(dist, bf.from_partition(part))
    optimal = (H(initial), initial)

    queue = deque([part])

    checked = set()

    while queue:  # pragma: no branch
        part = queue.popleft()

        checked.add(part)

        d = insert_rvf(dist, bf.from_partition(part))

        if np.isclose(B(d), 0):
            h = H(d)

            if h <= optimal[0]:
                optimal = (h, d)

            if np.isclose(h, optimal_b):
                break

            new_parts = [
                frozenset([p for p in part if p not in pair] + [pair[0] | pair[1]]) for pair in combinations(part, 2)
            ]
            new_parts = sorted((part for part in new_parts if part not in checked), key=lambda p: sorted(map(len, p)))
            queue.extendleft(new_parts)

    return optimal[1]


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

    d = functional_markov_chain(dist, rvs, crvs)
    return entropy(d, [dist.outcome_length()])
