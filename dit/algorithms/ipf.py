"""
Iterative Proportional Fitting (IPF) for maximum entropy distributions.

IPF (also known as iterative scaling, the RAS algorithm, or matrix raking) is
the classic algorithm used in reconstructability analysis and log-linear
modeling to compute the maximum entropy distribution consistent with a set of
marginal constraints. Starting from the uniform distribution, it cyclically
rescales the working distribution so that each constrained marginal matches the
data, iterating until convergence. For decomposable (acyclic) structures it
converges in a single cycle; for cyclic structures it iterates.
"""

import numpy as np

from ..helpers import parse_rvs
from .optutil import prepare_dist

__all__ = ("ipf_dist",)


def ipf_dist(dist, rvs, tol=1e-10, maxiter=10000, sparse=True, return_status=False):
    """
    Return the maximum entropy distribution consistent with the marginals from
    `dist` specified in `rvs`, computed via Iterative Proportional Fitting.

    Parameters
    ----------
    dist : Distribution
        The distribution whose marginals should be matched.
    rvs : list of lists
        The marginals from `dist` to constrain. Each inner list specifies a set
        of random variables (a marginal "projection") to hold fixed, e.g.
        ``[[0, 1], [1, 2]]`` for the structure ``AB:BC``.
    tol : float
        The convergence tolerance. Iteration stops once the largest deviation
        of any constrained marginal from its target falls below `tol`.
    maxiter : int
        The maximum number of IPF cycles to perform.
    sparse : bool
        Whether the returned distribution should be sparse or dense.
    return_status : bool
        If True, return a ``(me, converged)`` tuple, where `converged` is True
        iff the final marginal deviation fell below `tol` within `maxiter`
        cycles. IPF converges only linearly on cyclic structures with induced
        structural zeros, so callers that need an exact answer should check
        this flag and fall back to a convex optimizer when it is False.

    Returns
    -------
    me : Distribution
        The maximum entropy distribution.
    converged : bool
        Only returned when `return_status` is True (see above).
    """
    dist = prepare_dist(dist)

    parse = lambda rv: parse_rvs(dist, rv, unique=True, sort=True)[1]
    groups = [tuple(parse(rv)) for rv in rvs]

    shape = tuple(len(a) for a in dist.alphabet)
    n_variables = len(shape)

    data = dist.pmf.reshape(shape)

    # Precompute, for each constrained marginal, the target marginal from the
    # data and the complement axes summed over to form that marginal.
    complements = [tuple(i for i in range(n_variables) if i not in g) for g in groups]
    targets = [data.sum(axis=comp, keepdims=True) for comp in complements]

    # Start from the uniform distribution over the full sample space.
    q = np.full(shape, 1.0 / data.size)

    converged = True
    for _ in range(maxiter):
        for target, comp in zip(targets, complements, strict=True):
            current = q.sum(axis=comp, keepdims=True)
            # A zero target marginal forces the contributing cells to zero; a
            # zero current marginal leaves those (already zero) cells untouched.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(current > 0, target / current, 0.0)
            q = q * ratio

        deviation = max(
            (
                np.abs(q.sum(axis=comp, keepdims=True) - target).max()
                for target, comp in zip(targets, complements, strict=True)
            ),
            default=0.0,
        )
        if deviation < tol:
            break
    else:
        # Exhausted maxiter without the deviation ever dropping below tol.
        converged = False

    total = q.sum()
    if total > 0:
        q = q / total

    new_dist = dist.copy()
    new_dist.pmf = q.ravel()
    if sparse:
        new_dist.make_sparse()

    names = dist.get_rv_names()
    if names is not None:
        new_dist.set_rv_names(names)

    if return_status:
        return new_dist, converged
    return new_dist
