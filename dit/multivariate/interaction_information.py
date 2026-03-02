"""
The interaction information is a form of multivariate information.
"""

import numpy as np

from ..helpers import normalize_rvs
from ..utils import unitful
from .coinformation import coinformation

__all__ = ("interaction_information",)


@unitful
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
        Deprecated. Kept for signature compatibility.

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
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs)

    II = (-1) ** len(rvs) * coinformation(dist, rvs, crvs)

    if np.isclose(II, 0):
        II = 0.0

    return II
