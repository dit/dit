"""
The total correlation, aka the multi-information or the integration.
"""

from ..helpers import normalize_rvs
from ..shannon import conditional_entropy as H
from ..utils import unitful


@unitful
def total_correlation(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Computes the total correlation, also known as either the multi-information
    or the integration.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the total correlation is calculated.
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
    T : float
        The total correlation.

    Examples
    --------
    >>> d = dit.example_dists.Xor()
    >>> dit.multivariate.total_correlation(d)
    1.0
    >>> dit.multivariate.total_correlation(d, rvs=[[0], [1]])
    0.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    one = sum([H(dist, rv, crvs, rv_mode=rv_mode) for rv in rvs])
    two = H(dist, set().union(*rvs), crvs, rv_mode=rv_mode)
    T = one - two

    return T
