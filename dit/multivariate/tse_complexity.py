"""
The TSE Complexity.
"""
from __future__ import division

from itertools import combinations

from ..shannon import conditional_entropy as H
from ..helpers import normalize_rvs
from ..math.misc import combinations as nCk

def tse_complexity(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Calculates the TSE complexity.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the TSE complexity is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the TSE complexity
        between. If None, then the TSE complexity is calculated over all random
        variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    TSE : float
        The TSE complexity.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    joint = H(dist, set().union(*rvs), crvs, rv_mode=rv_mode)
    N = len(rvs)

    def sub_entropies(k):
        """
        Compute the average entropy of all subsets of `rvs` of size `k`.
        """
        sub_rvs = (set().union(*rv) for rv in combinations(rvs, k))
        subH = sum(H(dist, rv, crvs, rv_mode=rv_mode) for rv in sub_rvs)
        subH /= nCk(N, k)
        return subH

    TSE = sum(sub_entropies(k) - k/N * joint for k in range(1, N))

    return TSE
