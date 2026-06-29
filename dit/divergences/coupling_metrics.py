"""
Information-theoretic couplings: joint distributions with fixed marginals.

These differ from optimal-transport couplings (see
:func:`~dit.divergences.earth_movers_distance.earth_movers_distance`), which
minimize expected ground-metric cost.
"""

import numpy as np
from boltons.iterutils import pairwise

from ..helpers import normalize_rvs
from ..shannon import conditional_entropy as H
from ..utils import unitful

__all__ = (
    "coupling_metric",
    "coupling_min_residual_entropy",
    "max_caekl_coupling",
    "max_dual_total_correlation_coupling",
    "max_total_correlation_coupling",
    "min_residual_entropy_coupling",
)


def _coupling_problem(dists):
    """
    Build the tensor-product scaffold and per-marginal RV index groups.

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.

    Returns
    -------
    product_dist : Distribution
        Independent product used as the optimization scaffold.
    dist_ids : list of list of int
        RV index groups, one per marginal.
    """
    d = dists[0]
    for d2 in dists[1:]:
        d = d.__matmul__(d2)

    lengths = [0] + [len(dist.rvs) for dist in dists]
    dist_ids = [list(range(a, b)) for a, b in pairwise(np.cumsum(lengths))]

    return d, dist_ids


def _optimize_coupling(dists, optimizer_name, *, niter=50):
    """
    Return a joint distribution optimizing ``optimizer_name`` subject to marginals.

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.
    optimizer_name : str
        Name of a :class:`~dit.algorithms.distribution_optimizers.BaseDistOptimizer`
        subclass in ``dit.algorithms.distribution_optimizers``.
    niter : int
        Number of basin-hopping iterations.

    Returns
    -------
    dist : Distribution
        Optimized coupling.
    dist_ids : list of list of int
        RV index groups for each marginal.
    """
    from ..algorithms import distribution_optimizers as do
    from ..multivariate import residual_entropy as multivariate_residual_entropy

    OptimizerClass = getattr(do, optimizer_name)
    product_dist, dist_ids = _coupling_problem(dists)

    if optimizer_name == "MinResidualEntropyOptimizer":
        meo = do.MinEntOptimizer(product_dist, dist_ids)
        meo.optimize(niter=niter)
        me_dist = meo.construct_dist()

        opt = OptimizerClass(product_dist, dist_ids)
        opt.optimize(x0=meo._optima.copy(), niter=niter)
        re_dist = opt.construct_dist()

        if multivariate_residual_entropy(re_dist, rvs=dist_ids) <= multivariate_residual_entropy(me_dist, rvs=dist_ids):
            return re_dist, dist_ids
        return me_dist, dist_ids

    opt = OptimizerClass(product_dist, dist_ids)
    opt.optimize(niter=niter)
    return opt.construct_dist(), dist_ids


@unitful
def residual_entropy(dist, rvs=None, crvs=None, p=1.0):
    """
    Compute the residual entropy with an optional p-norm aggregation.

    This local helper supports the legacy :func:`coupling_metric` scalar, which
    uses a p-norm over per-variable conditional entropies. For the standard
    (L1) residual entropy used elsewhere in ``dit``, see
    :func:`dit.multivariate.residual_entropy`.

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
    Residual entropy of the minimum-entropy coupling with the given marginals.

    .. note::

        This uses :class:`~dit.algorithms.distribution_optimizers.MinEntOptimizer`
        (minimum joint entropy), not a direct minimization of residual entropy.
        For the latter, use :func:`coupling_min_residual_entropy` or
        :func:`min_residual_entropy_coupling`.

    Parameters
    ----------
    dists : list of Distribution
        The distributions to consider as marginals
    p : float
        The p-norm used when evaluating residual entropy on the coupling.

    Returns
    -------
    cm : float
        The residual entropy of the minimum joint-entropy coupling.
    """
    od, dist_ids = _optimize_coupling(dists, "MinEntOptimizer")
    return residual_entropy(od, rvs=dist_ids, p=p)


def min_residual_entropy_coupling(dists, *, niter=50):
    """
    Coupling with minimal residual entropy (variation of information).

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.
    niter : int
        Number of basin-hopping iterations.

    Returns
    -------
    dist : Distribution
        A joint distribution with the prescribed marginals and approximately
        minimal residual entropy.
    """
    dist, _ = _optimize_coupling(dists, "MinResidualEntropyOptimizer", niter=niter)
    return dist


def max_total_correlation_coupling(dists, *, niter=50):
    """
    Coupling with maximal total correlation (multi-information).

    With fixed marginals, this is equivalent to minimum joint entropy.

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.
    niter : int
        Number of basin-hopping iterations.

    Returns
    -------
    dist : Distribution
        A joint distribution with the prescribed marginals and approximately
        maximal total correlation.
    """
    dist, _ = _optimize_coupling(dists, "MinEntOptimizer", niter=niter)
    return dist


def max_dual_total_correlation_coupling(dists, *, niter=50):
    """
    Coupling with maximal dual total correlation (binding information).

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.
    niter : int
        Number of basin-hopping iterations.

    Returns
    -------
    dist : Distribution
        A joint distribution with the prescribed marginals and approximately
        maximal dual total correlation.
    """
    dist, _ = _optimize_coupling(dists, "MaxDualTotalCorrelationOptimizer", niter=niter)
    return dist


def max_caekl_coupling(dists, *, niter=50):
    """
    Coupling with maximal CAEKL mutual information.

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.
    niter : int
        Number of basin-hopping iterations.

    Returns
    -------
    dist : Distribution
        A joint distribution with the prescribed marginals and approximately
        maximal CAEKL mutual information.
    """
    dist, _ = _optimize_coupling(dists, "MaxCAEKLMutualInformationOptimizer", niter=niter)
    return dist


@unitful
def coupling_min_residual_entropy(dists, *, niter=25):
    """
    Minimum residual entropy over couplings with the given marginals.

    Parameters
    ----------
    dists : list of Distribution
        Marginal distributions to couple.
    niter : int
        Number of basin-hopping iterations.

    Returns
    -------
    R : float
        The residual entropy of :func:`min_residual_entropy_coupling`.
    """
    dist, dist_ids = _optimize_coupling(dists, "MinResidualEntropyOptimizer", niter=niter)
    from ..multivariate import residual_entropy as multivariate_residual_entropy

    return multivariate_residual_entropy(dist, rvs=dist_ids)
