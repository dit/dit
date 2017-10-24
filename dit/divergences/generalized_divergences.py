"""
The Kullback-Leibler divergence.
"""

import numpy as np

from .cross_entropy import get_pmfs_like
from ..helpers import normalize_rvs
from ..utils import flatten

from .kullback_leibler_divergence import kullback_leibler_divergence

__all__ = ('double_power_sum',
           'hellinger_sum',
           'alpha_divergence',
           'hellinger_divergence',
           'renyi_divergence',
           'tsallis_divergence',
          )

### References for Divergence Formulas ###
## http://arxiv.org/pdf/1105.3259v1.pdf
## http://arxiv.org/pdf/1206.2459.pdf
## http://mitran-lab.amath.unc.edu:8082/subversion/grants/Proposals/2013/DOE-DataCentric/biblio/LieseVajdaDivergencesInforTheory.pdf
## Crooks: http://threeplusone.com/on_information.pdf

def double_power_sum(dist1, dist2, exp1=1, exp2=1, rvs=None, crvs=None,
                     rv_mode=None):
    """
    A common generalization of the sums needed to compute the Hellinger
    and alpha divergences below.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the sum.
        The second distribution in the sum.
    exp1 : float, 1
        First exponent used in the power sum.
    exp2 : float, 1
        Second exponent used in the power sum.
    rvs : list, None
        The indexes of the random variable used to calculate the sum.
        If None, then the sum is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The specified sum between `dist1` and `dist2`.

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
    div = np.nansum(np.power(p1s, exp1) * np.power(q1s, exp2))

    if crvs:
        p2s, q2s = get_pmfs_like(dist1, dist2, crvs, rv_mode)
        div = np.nansum(np.power(p2s, exp1) * np.power(q2s, exp2))

    return div

def hellinger_sum(dist1, dist2, alpha=1., rvs=None, crvs=None, rv_mode=None):
    """
    The Hellinger sum/integral of `dist1` and `dist2`, used to define other
    divergences.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the sum.
    dist2 : Distribution
        The second distribution in the sum.
    alpha : float, 1
        Exponent used in the sum.
    rvs : list, None
        The indexes of the random variable used to calculate the sum.
        If None, then the sum is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The Hellinger sum between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.

    """

    return double_power_sum(dist1, dist2, alpha, 1.-alpha,
                            rvs=rvs, crvs=crvs, rv_mode=rv_mode)

def hellinger_divergence(dist1, dist2, alpha=1., rvs=None, crvs=None,
                         rv_mode=None):
    # http://mitran-lab.amath.unc.edu:8082/subversion/grants/Proposals/2013/DOE-DataCentric/biblio/LieseVajdaDivergencesInforTheory.pdf
    """
    The Hellinger divergence of `dist1` and `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the Hellinger divergence.
    dist2 : Distribution
        The second distribution in the Hellinger divergence.
    alpha : float, 1
        The divergence is a one parameter family in alpha.
    rvs : list, None
        The indexes of the random variable used to calculate the
        Hellinger divergence between. If None, then the Hellinger
        divergence is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The Hellinger divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.

    """

    if alpha == 1:
        return kullback_leibler_divergence(dist1, dist2, rvs=rvs,
                                           crvs=crvs, rv_mode=rv_mode)
    s = hellinger_sum(dist1, dist2, rvs=rvs, alpha=alpha,
                      crvs=crvs, rv_mode=rv_mode)
    return (s-1.)/(alpha-1.)

def tsallis_divergence(dist1, dist2, alpha=1., rvs=None, crvs=None,
                       rv_mode=None):
    """
    The Tsallis divergence of `dist1` and `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the Tsallis divergence.
    dist2 : Distribution
        The second distribution in the Tsallis divergence.
    alpha : float, 1
        The divergence is a one parameter family in alpha.
    rvs : list, None
        The indexes of the random variable used to calculate the
        Tsallis divergence between. If None, then the Tsallis
        divergence is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The Tsallis divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.

    """

    # D_T = (D_alpha -1) / (alpha-1)
    if alpha == 1:
        div = kullback_leibler_divergence(dist1, dist2, rvs=rvs,
                                          crvs=crvs, rv_mode=rv_mode)
    else:
        s = hellinger_sum(dist1, dist2, rvs=rvs, alpha=alpha,
                          crvs=crvs, rv_mode=rv_mode)
        div = (s - 1.) /(alpha - 1.)
    return div

def renyi_divergence(dist1, dist2, alpha=1., rvs=None, crvs=None, rv_mode=None):
    """
    The Renyi divergence of `dist1` and `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the Renyi divergence.
    dist2 : Distribution
        The second distribution in the Renyi divergence.
    alpha : float, 1
        The divergence is a one parameter family in alpha.
    rvs : list, None
        The indexes of the random variable used to calculate the
        Renyi divergence between. If None, then the Renyi
        divergence is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The Renyi divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.

    """

    # D_R = log D_alpha / (alpha-1)
    if alpha == 1:
        div = kullback_leibler_divergence(dist1, dist2, rvs=rvs,
                                          crvs=crvs, rv_mode=rv_mode)
    else:
        s = hellinger_sum(dist1, dist2, rvs=rvs,
                          alpha=alpha, crvs=crvs, rv_mode=rv_mode)
        div = np.log2(s) / (alpha - 1.)
    return div

def alpha_divergence(dist1, dist2, alpha=1., rvs=None, crvs=None, rv_mode=None):
    """
    The alpha divergence of `dist1` and `dist2`, as used in Information
    Geometry. Note there is more than one inequivalent definition of
    "alpha divergence" in the literature, this one comes
    from http://en.wikipedia.org/wiki/Information_geometry .

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the alpha divergence.
    dist2 : Distribution
        The second distribution in the alpha divergence.
    alpha : float, 1
        The divergence is a one parameter family in alpha.
    rvs : list, None
        The indexes of the random variable used to calculate the
        alpha divergence between. If None, then the alpha
        divergence is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The alpha divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.

    """

    if alpha == 1:
        return kullback_leibler_divergence(dist1, dist2, rvs=rvs,
                                           crvs=crvs, rv_mode=rv_mode)
    if alpha == -1:
        return kullback_leibler_divergence(dist2, dist1, rvs=rvs,
                                           crvs=crvs, rv_mode=rv_mode)
    s = double_power_sum(dist1, dist2, (1.-alpha)/2, (1.+alpha)/2,
                         rvs=rvs, crvs=crvs, rv_mode=rv_mode)
    return 4*(1.-s)/(1.-alpha*alpha)

def f_divergence(dist1, dist2, f, rvs=None, crvs=None, rv_mode=None):
    """
    The Csiszar f-divergence of `dist1` and `dist2`. Note that it is typically
    more accurate to use a specialized divergence function when available
    due to roundoff errors and small probability effects.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the f-divergence.
    dist2 : Distribution
        The second distribution in the f-divergence.
    f : function
        The auxillary function defining the f-divergence
    rvs : list, None
        The indexes of the random variable used to calculate the
        f-divergence between. If None, then the
        f-divergence is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The f-divergence between `dist1` and `dist2`.

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
    vfunc = np.vectorize(f)
    div = np.nansum(vfunc(np.divide(p1s, q1s)) * q1s)

    if crvs:
        p2s, q2s = get_pmfs_like(dist1, dist2, crvs, rv_mode)
        div = np.nansum(vfunc(np.divide(p2s, q2s)) * q2s)

    return div

