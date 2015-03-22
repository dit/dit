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
    >>> generalized_cumulative_residual_entropy(uniform(-2, 3))
    1.6928786893420307
    >>> generalized_cumulative_residual_entropy(uniform(0, 5))
    1.6928786893420307
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
    >>> d1 = ScalarDistribution([1, 2, 3, 4, 5, 6], [1/6]*6)
    >>> d2 = ScalarDistribution([1, 2, 3, 4, 5, 100], [1/6]*6)
    >>> cumulative_residual_entropy(d1)
    2.0683182557028439
    >>> cumulative_residual_entropy(d2)
    22.672680046016705
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
        The distribution to compute the conditional cumulative residual entropy
        of.
    rv : list, None
        The possibly joint random variable to compute the conditional cumulative
        residual entropy of. If `None`, then all variables not in `crvs` are
        used.
    crvs : list, None
        The random variables to condition on. If `None`, nothing is conditioned
        on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    CCRE : ScalarDistribution
        The conditional cumulative residual entropy.

    Examples
    --------
    >>> from itertools import product
    >>> events = [ (a, b) for a, b, in product(range(5), range(5)) if a <= b ]
    >>> probs = [ 1/(5-a)/5 for a, b in events ]
    >>> d = Distribution(events, probs)
    >>> print(conditional_cumulative_residual_entropy(d, 1, [0]))
    Class:    ScalarDistribution
    Alphabet: (-0.0, 0.5, 0.91829583405448956, 1.3112781244591329, 1.6928786893420307)
    Base:     linear

    x                p(x)
    -0.0             0.2
    0.5              0.2
    0.918295834054   0.2
    1.31127812446    0.2
    1.69287868934    0.2
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
        The distribution to compute the conditional generalized cumulative
        residual entropy of.
    rv : list, None
        The possibly joint random variable to compute the conditional
        generalized cumulative residual entropy of. If `None`, then all
        variables not in `crvs` are used.
    crvs : list, None
        The random variables to condition on. If `None`, nothing is conditioned
        on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    CCRE : ScalarDistribution
        The conditional cumulative residual entropy.

    Examples
    --------
    >>> from itertools import product
    >>> events = [ (a-2, b-2) for a, b, in product(range(5), range(5)) if a <= b ]
    >>> probs = [ 1/(3-a)/5 for a, b in events ]
    >>> d = Distribution(events, probs)
    >>> print(conditional_generalized_cumulative_residual_entropy(d, 1, [0]))
    Class:    ScalarDistribution
    Alphabet: (-0.0, 0.5, 0.91829583405448956, 1.3112781244591329, 1.6928786893420307)
    Base:     linear

    x                p(x)
    -0.0             0.2
    0.5              0.2
    0.918295834054   0.2
    1.31127812446    0.2
    1.69287868934    0.2
    """
    if crvs is None:
        crvs = []
    mdist, cdists = dist.condition_on(crvs=crvs, rvs=[rv], rv_mode=rv_mode)
    cres = [generalized_cumulative_residual_entropy(cd, extract=True) for cd in cdists]
    ccre = SD(cres, mdist.pmf)
    return ccre
