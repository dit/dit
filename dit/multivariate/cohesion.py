"""
Cohesion, a generalization bridging the total correlation and the dual total
correlation.
"""

from itertools import combinations

from ..helpers import normalize_rvs
from ..math import multinomial
from ..shannon import conditional_entropy as H
from ..utils import unitful

__all__ = ("cohesion",)


@unitful
def cohesion(dist, k, rvs=None, crvs=None, rv_mode=None):
    """
    Computes the k-cohesion.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the cohesion is calculated.
    k : int, 1 <= k < N
        The order of the cohesion to compute.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    C_k : float
        The k-Cohesion.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs)

    one = sum(H(dist, set().union(*rv_), crvs) for rv_ in combinations(rvs, k))
    two = H(dist, set().union(*rvs), crvs)
    C = one - multinomial(len(rvs) - 1, [k - 1, len(rvs) - k]) * two

    return C
