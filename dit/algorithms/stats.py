"""
Some basic statistics for distributions with numerical outcomes.
"""

from __future__ import division

import numpy as np

from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..math.misc import is_number
from ..utils import flatten

svdvals = lambda m: np.linalg.svd(m, compute_uv=False)

def _numerical_test(dist):
    """
    Verifies that all outcomes are numbers.

    Parameters
    ----------
    dist : Distribution
        The distribution whose outcomes are to be checked.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    if not all(is_number(o) for o in flatten(dist.outcomes)):
        msg = "The outcomes of this distribution are not numerical"
        raise TypeError(msg)

def mean(dist):
    """
    Computes the mean of the distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to take the mean of.

    Returns
    -------
    means : ndarray
        The mean of each index of the outcomes.

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    _numerical_test(dist)

    outcomes, pmf = zip(*dist.zipped(mode='patoms'))
    outcomes = np.asarray(outcomes)
    pmf = np.asarray(pmf)
    return np.average(outcomes, axis=0, weights=pmf)

def central_moment(dist, n):
    """
    Computes the nth central moment of a distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to take the moment of.
    n : int
        Which moment to take.

    Returns
    -------
    moments : ndarray
        The `n`th central moment of each index of the outcomes.

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    mu = mean(dist)
    outcomes, pmf = zip(*dist.zipped(mode='patoms'))
    outcomes = np.asarray(outcomes)
    pmf = np.asarray(pmf)
    terms = np.asarray([(np.asarray(o)-mu)**n for o in outcomes])
    terms[np.isnan(terms)] = 0
    return np.average(terms, axis=0, weights=pmf)

def standard_moment(dist, n):
    """
    Computes the nth standard moment of a distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to take the moment of.
    n : int
        Which moment to take.

    Returns
    -------
    moments : ndarray
        The `n`th standard moment of each index of the outcomes.

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    return central_moment(dist, n)/standard_deviation(dist)**n

def standard_deviation(dist):
    """
    Compute the standard deviation of a distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to take the standard deviation of.

    Returns
    -------
    std : ndarray
        The standard deviation of each index of the outcomes.

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    return np.sqrt(central_moment(dist, 2))

def median(dist):
    """
    Compute the median of a distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the median of.

    Returns
    -------
    medians : ndarray
        The median of each index of the outcomes.

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    _numerical_test(dist)

    g = np.asarray(dist.outcomes[(dist.pmf.cumsum() > 0.5).argmax()])
    ge = np.asarray(dist.outcomes[(dist.pmf.cumsum() >= 0.5).argmax()])
    return (g+ge)/2

def mode(dist):
    """
    Compute the modes of a distribution.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the modes of.

    Returns
    -------
    modes : [ndarray]
        A list of arrays, one for each index of the outcomes. Each array
        contains the modes of that index.

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    _numerical_test(dist)

    try:
        dists = [dist.marginal([i]) for i in range(dist.outcome_length())]
    except AttributeError:
        dists = [dist]

    modes = [np.asarray(d.outcomes)[d.pmf == d.pmf.max()] for d in dists]
    modes = [m.flatten() for m in modes]
    return modes

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
