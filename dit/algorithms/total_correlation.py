"""
The total correlation, aka the multi-information or the integration.
"""

from ..helpers import normalize_rvs
from .shannon import conditional_entropy as H

def total_correlation(dist, rvs=None, crvs=None, rv_names=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution from which the total correlation is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the total
        correlation. If None, then the total correlation is calculated
        over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.
    rv_names : bool, None
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    T : float
        The total correlation

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution.
    """
    rvs, crvs, rv_names = normalize_rvs(dist, rvs, crvs, rv_names)

    one = sum([ H(dist, rv, crvs, rv_names) for rv in rvs ])
    two = H(dist, set().union(*rvs), crvs, rv_names)
    T = one - two

    return T
