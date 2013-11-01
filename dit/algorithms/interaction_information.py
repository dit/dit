"""
The interaction information is a form of multivariate information.
"""

from ..helpers import normalize_rvs

from .coinformation import coinformation

def interaction_information(dist, rvs=None, crvs=None, rv_names=None):
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
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    II : float
        The interaction information.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution.
    """
    rvs, crvs, rv_names = normalize_rvs(dist, rvs, crvs, rv_names)

    II = (-1)**len(rvs) * coinformation(dist, rvs, crvs, rv_names)

    return II
