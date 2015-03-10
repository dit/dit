"""
The cross entropy.
"""

import numpy as np

from ..exceptions import ditException, InvalidOutcome
from ..helpers import normalize_rvs
from ..utils import flatten

__all__ = ('cross_entropy',
          )

def get_prob(d, o):
    """
    Get the probability of `o`, if it's not in the sample space return 0.
    """
    try:
        p = d[o]
    except InvalidOutcome:
        p = 0
    return p

def get_pmfs_like(d1, d2, rvs, rv_mode=None):
    """
    """
    dp = d1.marginal(rvs, rv_mode)
    dq = d2.marginal(rvs, rv_mode)
    ps = dp.pmf
    qs = np.asarray([ get_prob(dq, o) for o in dp.outcomes ])
    return ps, qs

def cross_entropy(dist1, dist2, rvs=None, crvs=None, rv_mode=None):
    """
    """
    rvs, crvs, rv_mode = normalize_rvs(dist1, rvs, crvs, rv_mode)
    rvs, crvs = list(flatten(rvs)), list(flatten(crvs))
    normalize_rvs(dist2, rvs, crvs, rv_mode)

    p1s, q1s = get_pmfs_like(dist1, dist2, rvs+crvs, rv_mode)
    xH = -np.nansum(p1s * np.log2(q1s))
    
    if crvs:
        p2s, q2s = get_pmfs_like(dist1, dist2, crvs, rv_mode)
        xH2 = -np.nansum(p2s * np.log2(q2s))
        xH -= xH2

    return xH
