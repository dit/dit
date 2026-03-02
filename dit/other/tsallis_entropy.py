"""
Tsallis Entropy.
"""

import numpy as np

from ..helpers import normalize_rvs
from ..multivariate import entropy
from ..utils import flatten

__all__ = ("tsallis_entropy",)


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
        Deprecated. Kept for signature compatibility.

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
        rvs = list(flatten(normalize_rvs(dist, rvs, None)[0]))
        dist = dist.marginal(rvs)

    pmf = dist.pmf

    S_q = entropy(dist) / np.log2(np.e) if order == 1 else 1 / (order - 1) * (1 - (pmf**order).sum())

    return S_q
