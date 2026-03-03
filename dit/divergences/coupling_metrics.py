"""
Metrics related to minimum entropy distributions with fixed marginals.
"""

import numpy as np
from boltons.iterutils import pairwise

from ..algorithms.distribution_optimizers import MinEntOptimizer
from ..helpers import normalize_rvs
from ..multivariate import entropy as H
from ..utils import unitful

__all__ = ("coupling_metric",)


@unitful
def residual_entropy(dist, rvs=None, crvs=None, p=1.0):
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
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    others = lambda rv, rvs: set(set().union(*rvs)) - set(rv)

    R = sum(H(dist, rv, others(rv, rvs).union(crvs)) ** p for rv in rvs) ** (1 / p)

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
