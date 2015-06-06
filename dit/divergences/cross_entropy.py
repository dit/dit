"""
The cross entropy.
"""

import numpy as np

from ..exceptions import InvalidOutcome
from ..helpers import normalize_rvs
from ..utils import flatten

__all__ = ('cross_entropy',
          )

def get_prob(d, o):
    """
    Get the probability of `o`, if it's not in the sample space return 0.

    Parameters
    ----------
    d : Distribution
        The distribution to get the outcomes of.
    o : object
        The event to get the probability of.

    Returns
    -------
    p : float
        The probability of `o`.
    """
    try:
        p = d[o]
    except InvalidOutcome:
        p = 0
    return p

def get_pmfs_like(d1, d2, rvs, rv_mode=None):
    """
    Get the pmf from `d1` for `rvs`, and the pmf from `d2` for the events in
    `d1`

    Parameters
    ----------
    d1 : Distribution
        The distribution to get the pmf for.
    d2 : Distribution
        The distribution to get the pmf for, with the outcomes from `d1`.
    rvs : list, None
        The random variables to get the pmf for.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    ps : ndarray
        The pmf of d1.
    qs : ndarray
        A matching pmf from d2.
    """
    dp = d1.marginal(rvs, rv_mode)
    dq = d2.marginal(rvs, rv_mode)
    ps = dp.pmf
    qs = np.asarray([get_prob(dq, o) for o in dp.outcomes])
    return ps, qs

def cross_entropy(dist1, dist2, rvs=None, crvs=None, rv_mode=None):
    """
    The cross entropy between `dist1` and `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the cross entropy.
    dist2 : Distribution
        The second distribution in the cross entropy.
    rvs : list, None
        The indexes of the random variable used to calculate the cross entropy
        between. If None, then the cross entropy is calculated over all random
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
    xh : float
        The cross entropy between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist1, rvs, crvs, rv_mode)
    rvs, crvs = list(flatten(rvs)), list(flatten(crvs))
    normalize_rvs(dist2, rvs, crvs, rv_mode)

    p1s, q1s = get_pmfs_like(dist1, dist2, rvs+crvs, rv_mode)
    xh = -np.nansum(p1s * np.log2(q1s))

    if crvs:
        p2s, q2s = get_pmfs_like(dist1, dist2, crvs, rv_mode)
        xh2 = -np.nansum(p2s * np.log2(q2s))
        xh -= xh2

    return xh
