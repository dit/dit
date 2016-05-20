"""
The interaction information is a form of multivariate information.
"""

from ..helpers import normalize_rvs

from .coinformation import coinformation
from ..math import close

def interaction_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Calculates the interaction information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the interaction information is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the interaction
        information between. If None, then the interaction information is
        calculated over all random variables.
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
    II : float
        The interaction information.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    II = (-1)**len(rvs) * coinformation(dist, rvs, crvs, rv_mode)

    if close(II, 0):
        II = 0.0

    return II
