"""
Renyi Entropy.
"""

import numpy as np

from ..helpers import normalize_rvs
from ..utils import flatten
from ..multivariate import entropy


__all__ = (
    'renyi_entropy',
)


def renyi_entropy(dist, order, rvs=None, rv_mode=None):
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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

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
        rvs = list(flatten(normalize_rvs(dist, rvs, None, rv_mode)[0]))
        dist = dist.marginal(rvs, rv_mode)

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


def sibson_mutual_information_pmf(p_xy, order):
    """
    Compute the Sibson mutual information of order `order`.

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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

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
    if order <= 0:
        msg = "`order` must be a non-zero positive real number"
        raise ValueError(msg)

    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    if order == 1:
        H_a = p_xy * np.log(p_xy / (p_x * p_y))
    elif order == np.inf:
        H_a = -np.log2(p_xy.max())
    else:
        H_a = 1 / (1 - order) * np.log2((p_xy**order).sum())

    return H_a


def sibson_mutual_information(dist, order, rvs=None, rv_mode=None):
    """
    Compute the Sibson mutual information of order `order`.

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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

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
        rvs = list(flatten(normalize_rvs(dist, rvs, None, rv_mode)[0]))
        dist = dist.marginal(rvs, rv_mode)

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
