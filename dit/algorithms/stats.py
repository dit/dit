"""
Some basic statistics for distributions with numerical outcomes.
"""

from __future__ import division

import numpy as np

from ..math.misc import is_number
from ..utils import flatten


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
