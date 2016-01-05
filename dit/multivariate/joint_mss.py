"""
Compute the entropy of the joint minimal sufficient statistic.
"""

from ..algorithms.minimal_sufficient_statistic import insert_joint_mss
from .entropy import entropy

def joint_mss_entropy(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Compute the entropy of the join of the minimal sufficent statistic of each variable about the others.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the joint minimal sufficient statistic is computed.
    rvs : list, None
        The random variables to compute the joint minimal sufficient statistic of. If None, all random variables are used.
    crvs : list, None
        The random variables to condition the joint minimal sufficient statistic on. If None, then no random variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    """
    d = insert_joint_mss(dist, -1, rvs, rv_mode)

    M = entropy(d, [d.outcome_length() - 1], crvs, rv_mode)
    return M
