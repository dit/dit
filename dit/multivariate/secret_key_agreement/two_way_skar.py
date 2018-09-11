"""
"""

import numpy as np

from .no_communication import no_communication_skar
from .intrinsic_mutual_informations import intrinsic_total_correlation as intrinsic_mutual_information
from .skar_lower_bounds import necessary_intrinsic_mutual_information
from .minimal_intrinsic_mutual_informations import minimal_intrinsic_total_correlation as minimal_intrinsic_mutual_information
from .interactive_skar import interactive_skar
from .two_part_intrinsic_mutual_informations import two_part_intrinsic_total_correlation as two_part_intrinsic_mutual_information


__all__ = [
    'two_way_skar',
    'two_way_skar_bounds',
]


def _two_way_skar_bounds_iter(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Iteratively compute tighter bounds on the two way secret key agreement rate.

    Parameters
    -----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable
        The indices to consider as X (Alice) and Y (Bob).
    crvs : iterable
        The indices to consider as Z (Eve).
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Yields
    ------
    lower : float
        The lower bound.
    upper : float
        The upper bound.
    """
    bound_func = lambda i, x, y: [x, y][i%2]

    lower = no_communication_skar(dist, rvs[0], rvs[1], crvs, rv_mode=rv_mode)
    upper = intrinsic_mutual_information(dist, rvs, crvs, rv_mode=rv_mode)
    yield lower, upper
    new_lower = necessary_intrinsic_mutual_information(dist, rvs, crvs, rv_mode=rv_mode)
    lower = max([lower, new_lower])
    yield lower, upper
    new_upper = minimal_intrinsic_mutual_information(dist, rvs, crvs, rv_mode=rv_mode)
    upper = min([upper, new_upper])
    yield lower, upper
    new_lower = interactive_skar(dist, rvs, crvs, bound_func=bound_func, rounds=2, rv_mode=rv_mode)
    lower = max([lower, new_lower])
    yield lower, upper
    new_lower = interactive_skar(dist, rvs, crvs, bound_func=bound_func, rounds=3, rv_mode=rv_mode)
    lower = max([lower, new_lower])
    yield lower, upper
    new_lower = interactive_skar(dist, rvs, crvs, bound_func=bound_func, rounds=4, rv_mode=rv_mode)
    lower = max([lower, new_lower])
    yield lower, upper
    new_upper = two_part_intrinsic_mutual_information(dist, rvs, crvs, bound_j=2, bound_u=2, bound_v=2, rv_mode=rv_mode)
    upper = min([upper, new_upper])
    yield lower, upper


def two_way_skar_bounds(dist, rvs, crvs, rv_mode=None):
    """
    Iteratively compute tighter bounds on the two way secret key agreement rate.

    Parameters
    -----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable
        The indices to consider as X (Alice) and Y (Bob).
    crvs : iterable
        The indices to consider as Z (Eve).
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    lower : float
        The best lower bound.
    upper : float
        The best upper bound.
    """
    for lower, upper in _two_way_skar_bounds_iter(dist, rvs, crvs, rv_mode):
        if np.isclose(lower, upper):
            return lower, upper
    return lower, upper


def two_way_skar(dist, rvs, crvs, rv_mode=None):
    """
    Compute the two way secret key agreement rate. Returns nan if it can not be
    definitively determined.

    Parameters
    -----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable
        The indices to consider as X (Alice) and Y (Bob).
    crvs : iterable
        The indices to consider as Z (Eve).
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    skar : float
        The two way secret key agreement rate.
    """
    lower, upper = two_way_skar_bounds(dist, rvs, crvs, rv_mode)
    if np.isclose(lower, upper):
        return lower
    else:
        return np.nan
