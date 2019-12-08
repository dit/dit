# -*- coding: utf-8 -*-

"""
Secret Key Agreement Rate when communication is not permitted.
"""

from .. import gk_common_information
from ...utils import unitful


__all__ = [
    'no_communication_skar',
]


@unitful
def no_communication_skar(dist, rv_x, rv_y, rv_z, rv_mode=None):
    """
    The rate at which X and Y can agree upon a key with Z eavesdropping,
    and no public communication.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rv_x : iterable
        The indices to consider as the X variable, Alice.
    rv_y : iterable
        The indices to consider as the Y variable, Bob.
    rv_z : iterable
        The indices to consider as the Z variable, Eve.
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
        The no-communication secret key agreement rate.
    """
    return gk_common_information(dist, [rv_x, rv_y], rv_z, rv_mode=rv_mode)
