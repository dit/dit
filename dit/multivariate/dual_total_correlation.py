"""
The dual total correlation and variation of information.
"""

from ..shannon import conditional_entropy as H
from ..helpers import normalize_rvs

__all__ = ('binding_information',
           'dual_total_correlation',
           'independent_information',
           'residual_entropy',
           'variation_of_information',
          )

def dual_total_correlation(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Calculates the dual total correlation, also known as the binding
    information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the dual total correlation is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the binding
        information. If None, then the dual total correlation is calculated
        over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    B : float
        The dual total correlation.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    others = lambda rv, rvs: set(set().union(*rvs)) - set(rv)

    one = H(dist, set().union(*rvs), crvs, rv_mode=rv_mode)
    two = sum(H(dist, rv, others(rv, rvs).union(crvs), rv_mode=rv_mode)
              for rv in rvs)
    B = one - two

    return B


def residual_entropy(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Compute the residual entropy.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the residual entropy is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the residual
        entropy. If None, then the total correlation is calculated
        over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    R : float
        The residual entropy.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    others = lambda rv, rvs: set(set().union(*rvs)) - set(rv)

    R = sum(H(dist, rv, others(rv, rvs).union(crvs), rv_mode=rv_mode)
            for rv in rvs)

    return R


binding_information = dual_total_correlation

independent_information = variation_of_information = residual_entropy
