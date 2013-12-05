"""
The TSE Complexity.
"""
from __future__ import division

from itertools import combinations

from ..shannon import conditional_entropy as H
from ..helpers import normalize_rvs
from ..math.misc import combinations as nCk

def tse_complexity(dist, rvs=None, crvs=None, rv_names=None):
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
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    TSE : float
        The TSE complexity.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution.
    """
    rvs, crvs, rv_names = normalize_rvs(dist, rvs, crvs, rv_names)

    joint = H(dist, set().union(*rvs), crvs, rv_names)
    N = len(rvs)

    def sub_entropies(k):
        """
        Compute the average entropy of all subsets of `rvs` of size `k`.
        """
        sub_rvs = ( set().union(*rv) for rv in combinations(rvs, k) )
        subH = sum( H(dist, rv, crvs, rv_names) for rv in sub_rvs )
        subH /= nCk(N, k)
        return subH

    TSE = sum( sub_entropies(k) - k/N * joint for k in range(1, N) )

    return TSE
