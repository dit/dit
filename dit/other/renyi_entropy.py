"""
Renyi Entropy.
"""

import numpy as np

from ..helpers import normalize_rvs
from ..multivariate import entropy
from ..utils import flatten

__all__ = ("renyi_entropy",)


def renyi_entropy(dist, order, rvs=None):
    """
    Compute the Renyi entropy of order `order`.

    Parameters
    ----------
    dist : Distribution
        The distribution to take the Renyi entropy of.
    order : float >= 0
        The order of the Renyi entropy.
    rvs : list, None
        The indexes of the random variable used to calculate the Renyi entropy
        of. If None, then the Renyi entropy is calculated over all random
        variables.

    Returns
    -------
    H_a : float
        The Renyi entropy.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.
    ValueError
        Raised if `order` is not a non-negative float.

    """
    if order < 0:
        msg = "`order` must be a non-negative real number"
        raise ValueError(msg)

    if dist.is_joint and rvs is not None:
        rvs = list(flatten(normalize_rvs(dist, rvs, None)[0]))
        dist = dist.marginal(rvs)

    pmf = dist.pmf

    if order == 0:
        H_a = np.log2(pmf.size)
    elif order == 1:
        H_a = entropy(dist)
    elif order == np.inf:
        H_a = -np.log2(pmf.max())
    else:
        H_a = 1 / (1 - order) * np.log2((pmf**order).sum())

    return H_a
