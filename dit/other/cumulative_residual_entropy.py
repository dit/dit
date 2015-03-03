"""
The (generalized) cumulative residual entropy and conditional (generalized)
cumulative residual entropy.
"""

from six.moves import range # pylint: disable=redefined-builtin,import-error

import numpy as np

from .. import Distribution as D, ScalarDistribution as SD
from ..algorithms.stats import _numerical_test
from ..utils import pairwise

__all__ = ['cumulative_residual_entropy',
           'generalized_cumulative_residual_entropy',
           'conditional_cumulative_residual_entropy',
           'conditional_generalized_cumulative_residual_entropy',
          ]

def _cumulative_residual_entropy(dist, generalized=False):
    """
    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantagious properties.

    Parameters
    ----------
    dist : ScalarDistribution
        The distribution to compute the cumulative residual entropy of.
    generalized : bool
        Wheither to integrate from zero over the CDF or to integrate from zero
        over the CDF of the absolute value.

    Returns
    -------
    CRE : float
        The (generalized) cumulative residual entropy.

    Examples
    --------
    """
    _numerical_test(dist)
    eps = ((e if generalized else abs(e), p) for e, p in dist.zipped())
    events, probs = zip(*sorted(eps))
    cdf = dict((a, p) for a, p in zip(events, np.cumsum(probs)))
    terms = []
    for a, b in pairwise(events):
        pgx = cdf[a]
        term = (b-a)*pgx*np.log2(pgx)
        terms.append(term)
    return -np.nansum(terms)

def generalized_cumulative_residual_entropy(dist, extract=False):
    """
    The generalized cumulative residual entropy is a generalized from of the
    cumulative residual entropy. Rarther than integrating from 0 to infinty over
    the absolute value of the CDF.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the generalized cumulative residual entropy
        of each index for.
    extract : bool
        If True and `dist.outcome_length()` is 1, return the single GCRE value
        rather than a length-1 array.

    Returns
    -------
    GCREs : ndarray
        The generalized cumulative residual entropy for each index.

    Examples
    --------
    """
    if not dist.is_joint():
        return _cumulative_residual_entropy(dist, generalized=True)
    length = dist.outcome_length()
    margs = [SD.from_distribution(dist.marginal([i])) for i in range(length)]
    cres = np.array([_cumulative_residual_entropy(m, generalized=True) for m in margs])
    if len(cres) == 1 and extract:
        cres = cres[0]
    return cres

def cumulative_residual_entropy(dist, extract=False):
    """
    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several desirable properties including
    non-negativity.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the cumulative residual entropy of each
        index for.
    extract : bool
        If True and `dist.outcome_length()` is 1, return the single GCRE value
        rather than a length-1 array.

    Returns
    -------
    CREs : ndarray
        The cumulative residual entropy for each index.

    Examples
    --------
    """
    if not dist.is_joint():
        return _cumulative_residual_entropy(dist, generalized=False)
    es, ps = zip(*[(tuple(abs(ei) for ei in e), p) for e, p in dist.zipped()])
    abs_dist = D(es, ps)
    return generalized_cumulative_residual_entropy(abs_dist, extract)

def conditional_cumulative_residual_entropy(dist, rv, crvs=None, rv_mode=None):
    """
    Returns the conditional cumulative residual entropy.

    Parameters
    ----------
    dist : Distribution
    rv : list, None
    crvs : list, None

    Returns
    -------
    CCRE : ScalarDistribution
        The conditional cumulative residual entropy.

    Examples
    --------
    """
    if crvs is None:
        crvs = []
    mdist, cdists = dist.condition_on(crvs=crvs, rvs=[rv], rv_mode=rv_mode)
    cres = [cumulative_residual_entropy(cd, extract=True) for cd in cdists]
    ccre = SD(cres, mdist.pmf)
    return ccre

def conditional_generalized_cumulative_residual_entropy(dist, rv, crvs=None, rv_mode=None):
    """
    Returns the conditional cumulative residual entropy.

    Parameters
    ----------
    dist : Distribution
    rv : list, None
    crvs : list, None

    Returns
    -------
    CCRE : ScalarDistribution
        The conditional cumulative residual entropy.

    Examples
    --------
    """
    if crvs is None:
        crvs = []
    mdist, cdists = dist.condition_on(crvs=crvs, rvs=[rv], rv_mode=rv_mode)
    cres = [generalized_cumulative_residual_entropy(cd, extract=True) for cd in cdists]
    ccre = SD(cres, mdist.pmf)
    return ccre
