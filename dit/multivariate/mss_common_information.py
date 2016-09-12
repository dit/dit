"""
Compute the minimal sufficient statistic common information.
"""

from ..algorithms.minimal_sufficient_statistic import insert_joint_mss
from ..helpers import normalize_rvs
from .dual_total_correlation import dual_total_correlation
from .entropy import entropy
from ..math import close

def mss_common_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Compute the minimal sufficient statistic common information, which is the
    entropy of the join of the minimal sufficent statistic of each variable
    about the others.

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
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    dtc = dual_total_correlation(dist, rvs, crvs, rv_mode)
    ent = entropy(dist, rvs, crvs, rv_mode)
    if close(dtc, ent):
        return dtc

    d = insert_joint_mss(dist, -1, rvs, rv_mode)

    M = entropy(d, [d.outcome_length() - 1], crvs, rv_mode)
    return M
