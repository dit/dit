"""
Metrics related to minimum entropy distributions with fixed marginals.
"""

from __future__ import division

from boltons.iterutils import pairwise

import numpy as np

from ..algorithms.distribution_optimizers import MinEntOptimizer
from ..helpers import normalize_rvs
from ..multivariate import entropy as H
from ..utils import unitful

__all__ = [
    'coupling_metric',
]


@unitful
def residual_entropy(dist, rvs=None, crvs=None, p=1.0, rv_mode=None):
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
    p : float
        The p-norm to utilize
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

    R = sum(H(dist, rv, others(rv, rvs).union(crvs), rv_mode=rv_mode)**p
            for rv in rvs)**(1/p)

    return R


def coupling_metric(dists, p=1.0):
    """
    Compute the minimum possible residual entropy of a joint distribution
    with `dists` as marginals.

    Parameters
    ----------
    dists : list of Distributions
        The distributions to consider as marginals
    p : float
        The p-norm.

    Returns
    -------
    cm : float
        The minimum residual entropy over all possible distributions with
        `dists` as marginals.
    """
    d = dists[0]
    for d2 in dists[1:]:
        d = d.__matmul__(d2)

    lengths = [0] + [len(dist.rvs) for dist in dists]
    dist_ids = [list(range(a, b)) for a, b in pairwise(np.cumsum(lengths))]

    meo = MinEntOptimizer(d, dist_ids)
    meo.optimize(niter=25)

    od = meo.construct_dist()
    re = residual_entropy(od, rvs=dist_ids, p=p)

    return re
