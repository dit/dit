"""
Tsallis Entropy
"""

from __future__ import division

import numpy as np

from ..helpers import normalize_rvs
from ..utils import flatten
from ..multivariate import entropy

__all__ = ('tsallis_entropy',
          )

def tsallis_entropy(dist, order, rvs=None, rv_mode=None):
    """
    Compute the Tsallis entropy of order `order`.

    Parameters
    ----------
    dist : Distribution
        The distribution to take the Tsallis entropy of.
    order : float >= 0
        The order of the Tsallis entropy.
    rvs : list, None
        The indexes of the random variable used to calculate the Tsallis entropy
        of. If None, then the Tsallis entropy is calculated over all random
        variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    S_q : float
        The Tsallis entropy.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.
    ValueError
        Raised if `order` is not a non-negative float.

    """
    if dist.is_joint and rvs is not None:
        rvs = list(flatten(normalize_rvs(dist, rvs, None, rv_mode)[0]))
        dist = dist.marginal(rvs, rv_mode)

    pmf = dist.pmf

    if order == 1:
        S_q = entropy(dist)/np.log2(np.e)
    else:
        S_q = 1/(order - 1) * (1 - (pmf**order).sum())

    return S_q
