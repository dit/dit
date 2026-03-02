"""
The (generalized) cumulative residual entropy and conditional (generalized)
cumulative residual entropy.
"""

import numpy as np
from boltons.iterutils import pairwise

from .. import Distribution
from ..helpers import numerical_test

__all__ = (
    "cumulative_residual_entropy",
    "generalized_cumulative_residual_entropy",
    "conditional_cumulative_residual_entropy",
    "conditional_generalized_cumulative_residual_entropy",
)


def _cumulative_residual_entropy(dist, generalized=False):
    """
    The cumulative residual entropy is an alternative to the Shannon
    differential entropy with several advantageous properties.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the cumulative residual entropy of.
    generalized : bool
        Whether to integrate from zero over the CDF or to integrate from zero
        over the CDF of the absolute value.

    Returns
    -------
    CRE : float
        The (generalized) cumulative residual entropy.

    Examples
    --------
    """
    numerical_test(dist)
    eps = ((e if generalized else abs(e), p) for e, p in dist.zipped())
    events, probs = zip(*sorted(eps), strict=True)
    cdf = {a: p for a, p in zip(events, np.cumsum(probs), strict=True)}
    terms = []
    for a, b in pairwise(events):
        pgx = cdf[a]
        term = (b - a) * pgx * np.log2(pgx)
        terms.append(term)
    return -np.nansum(terms)


def generalized_cumulative_residual_entropy(dist, extract=False):
    """
    The generalized cumulative residual entropy is a generalized from of the
    cumulative residual entropy. Rather than integrating from 0 to infinity over
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
    margs = [dist.marginal([i]) for i in range(length)]
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
    >>> d1 = Distribution([1, 2, 3, 4, 5, 6], [1/6]*6)
    >>> d2 = Distribution([1, 2, 3, 4, 5, 100], [1/6]*6)
    >>> cumulative_residual_entropy(d1)
    2.0683182557028439
    >>> cumulative_residual_entropy(d2)
    22.672680046016705
    """
    if not dist.is_joint():
        return _cumulative_residual_entropy(dist, generalized=False)
    # Build a distribution of absolute-valued outcomes
    pairs = []
    for e, p in dist.zipped():
        abs_e = tuple(abs(ei) for ei in e)
        pairs.append((abs_e, p))
    es, ps = zip(*pairs, strict=True)
    abs_dist = Distribution(list(es), list(ps))
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
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    CCRE : Distribution
        The conditional cumulative residual entropy.

    Examples
    --------
    >>> from itertools import product
    >>> events = [ (a, b) for a, b, in product(range(5), range(5)) if a <= b ]
    >>> probs = [ 1/(5-a)/5 for a, b in events ]
    >>> d = Distribution(events, probs)
    >>> print(conditional_cumulative_residual_entropy(d, 1, [0]))
    Class:    Distribution
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
    mdist, cdists = dist.condition_on(crvs=crvs, rvs=[rv])
    cres = [cumulative_residual_entropy(cd, extract=True) for cd in cdists]
    ccre = Distribution(cres, mdist.pmf)
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
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    CCRE : Distribution
        The conditional cumulative residual entropy.

    Examples
    --------
    >>> from itertools import product
    >>> events = [ (a-2, b-2) for a, b, in product(range(5), range(5)) if a <= b ]
    >>> probs = [ 1/(3-a)/5 for a, b in events ]
    >>> d = Distribution(events, probs)
    >>> print(conditional_generalized_cumulative_residual_entropy(d, 1, [0]))
    Class:    Distribution
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
    mdist, cdists = dist.condition_on(crvs=crvs, rvs=[rv])
    cres = [generalized_cumulative_residual_entropy(cd, extract=True) for cd in cdists]
    ccre = Distribution(cres, mdist.pmf)
    return ccre
