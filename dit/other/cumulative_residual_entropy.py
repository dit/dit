"""
The (generalized) cumulative residual entropy and conditional (generalized)
cumulative residual entropy.
"""

from functools import partial, update_wrapper

from six.moves import range # pylint: disable=redefined-builtin

import numpy as np

from .. import ScalarDistribution as SD
from ..algorithms.stats import _numerical_test
from ..helpers import normalize_rvs
from ..utils import pairwise

__all__ = ['cumulative_residual_entropy',
           'generalized_cumulative_residual_entropy',
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
    cdf = {a: p for a, p in zip(events, np.cumsum(probs))}
    terms = []
    for a, b in pairwise(events):
        pgx = cdf[a]
        term = (b-a)*pgx*np.log2(pgx)
        terms.append(term)
    return -np.nansum(terms)

def cumulative_residual_entropy(dist, generalized=False):
    """
    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantagious properties.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the cumulative residual entropy of each
        index for.
    generalized : bool
        Wheither to integrate from zero over the CDF of the absolute value
        or from negative infinity over the CDF.

    Returns
    -------
    CREs : ndarray
        The (generalized) cumulative residual entropy for each index.

    Examples
    --------
    """
    if not dist.is_joint():
        return _cumulative_residual_entropy(dist, generalized)
    length = dist.outcome_length()
    margs = [SD.from_distribution(dist.marginal([i])) for i in range(length)]
    cres = np.array([_cumulative_residual_entropy(m, generalized) for m in margs])
    return cres

def conditional_cumulative_residual_entropy(dist, rv, crvs=None, rv_mode=None, generalized=False):
    """
    Returns the conditional cumulative residual entropy.

    Parameters
    ----------
    dist : Distribution
    rv : list, None
    crvs : list, None
    generalized : bool
        Wheither to integrate from zero over the CDF of the absolute value
        or from negative infinity over the CDF.

    Returns
    -------
    CCRE : ScalarDistribution
        The conditional cumulative residual entropy.

    Examples
    --------
    """
    pass

generalized_cumulative_residual_entropy = partial(cumulative_residual_entropy, generalized=True)
update_wrapper(generalized_cumulative_residual_entropy, cumulative_residual_entropy)
