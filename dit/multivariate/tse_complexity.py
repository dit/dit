"""
The TSE Complexity.
"""

from itertools import combinations

from ..helpers import normalize_rvs
from ..math.misc import combinations as nCk
from ..shannon import conditional_entropy as H
from ..utils import unitful

__all__ = ("tse_complexity",)


@unitful
def tse_complexity(dist, rvs=None, crvs=None):
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
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    joint = H(dist, set().union(*rvs), crvs)
    N = len(rvs)

    def sub_entropies(k):
        """
        Compute the average entropy of all subsets of `rvs` of size `k`.
        """
        sub_rvs = (set().union(*rv) for rv in combinations(rvs, k))
        subH = sum(H(dist, rv, crvs) for rv in sub_rvs)
        subH /= nCk(N, k)
        return subH

    TSE = sum(sub_entropies(k) - k / N * joint for k in range(1, N))

    return TSE
