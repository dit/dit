"""
Cohesion, a generalization bridging the total correlation and the dual total
correlation.
"""

from itertools import combinations

from ..helpers import normalize_rvs
from ..math import multinomial
from ..shannon import conditional_entropy as H
from ..utils import unitful

__all__ = (
    'cohesion',
)


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
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

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
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    one = sum(H(dist, set().union(*rv_), crvs, rv_mode=rv_mode) for rv_ in combinations(rvs, k))
    two = H(dist, set().union(*rvs), crvs, rv_mode=rv_mode)
    C = one - multinomial(len(rvs) - 1, [k - 1, len(rvs) - k]) * two

    return C
