"""
Compute the maximum correlation:
    \rho(X:Y) = max_{f, g} E(f(X)g(Y))
"""

import numpy as np

from ..exceptions import ditException
from ..helpers import normalize_rvs


svdvals = lambda m: np.linalg.svd(m, compute_uv=False)


def conditional_maximum_correlation_pmf(pmf):
    """
    Compute the conditional maximum correlation from a 3-dimensional
    pmf. The maximum correlation is computed between the first two dimensions
    given the third.

    Parameters
    ----------
    pmf : ndarray
        The probability distribution.

    Returns
    -------
    rho_max : float
        The conditional maximum correlation.
    """
    pXYgZ = pmf / pmf.sum(axis=(0,1), keepdims=True)
    pXgZ = pXYgZ.sum(axis=1, keepdims=True)
    pYgZ = pXYgZ.sum(axis=0, keepdims=True)
    Q = np.where(pmf, pXYgZ / (np.sqrt(pXgZ)*np.sqrt(pYgZ)), 0)
    Q[np.isnan(Q)] = 0

    rho_max = max([ svdvals(np.squeeze(m))[1] for m in np.dsplit(Q, Q.shape[2]) ])

    return rho_max


def maximum_correlation_pmf(pXY):
    """
    Compute the maximum correlation from a 2-dimensional
    pmf. The maximum correlation is computed between the  two dimensions.

    Parameters
    ----------
    pmf : ndarray
        The probability distribution.

    Returns
    -------
    rho_max : float
        The maximum correlation.
    """
    pX = pXY.sum(axis=1, keepdims=True)
    pY = pXY.sum(axis=0, keepdims=True)
    Q = pXY / (np.sqrt(pX)*np.sqrt(pY))
    Q[np.isnan(Q)] = 0

    rho_max = svdvals(Q)[1]

    return rho_max


def maximum_correlation(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Compute the (conditional) maximum or Renyi correlation between two variables:

        rho_max = max_{f, g} rho(f(X,Z), g(Y,Z) | Z)

    Parameters
    ----------
    dist : Distribution
        The distribution for wich the maximum correlation is to computed.
    rvs : list, None; len(rvs) == 2
        A list of lists. Each inner list specifies the indexes of the random
        variables for which the maximum correlation is to be computed. If None,
        then all random variables are used, which is equivalent to passing
        `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to
        condition on. If None, then no variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    rho_max : float; -1 <= rho_max <= 1
        The conditional maximum correlation between `rvs` given `crvs`.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    if len(rvs) != 2:
        msg = 'Maximum correlation can only be computed for 2 variables, not {}.'.format(len(rvs))
        raise ditException(msg)

    if crvs:
        dist = dist.copy().coalesce(rvs + [crvs])
    else:
        dist = dist.copy().coalesce(rvs)

    dist.make_dense()
    pmf = dist.pmf.reshape(list(map(len, dist.alphabet)))

    if crvs:
        rho_max = conditional_maximum_correlation_pmf(pmf)
    else:
        rho_max = maximum_correlation_pmf(pmf)

    return rho_max
