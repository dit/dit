"""
Compute the minimal sufficient statistic common information.
"""

from copy import deepcopy

import numpy as np

from ...algorithms.minimal_sufficient_statistic import insert_joint_mss
from ...helpers import normalize_rvs
from ...utils import unitful
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy

__all__ = ("mss_common_information",)


@unitful
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
        Deprecated. Kept for signature compatibility.

    """
    dist = deepcopy(dist)
    dist.make_sparse()
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs)

    dtc = dual_total_correlation(dist, rvs, crvs)
    ent = entropy(dist, rvs, crvs)
    if np.isclose(dtc, ent):
        return dtc

    d = insert_joint_mss(dist, -1, rvs)

    M = entropy(d, [d.outcome_length() - 1], crvs)
    return M
