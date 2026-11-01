"""
I_IG, Niu & Quinn.

This measure is inspired by information geometry, and finds a point between
S0 - S1 - T and S1 - S0 - T such that the DKL between that point and the true
distribution is minimized. That minimal DKL is then the synergy.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ...exceptions import ditException
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_IG",)


def ig_synergy(dist, sources, target, fuzz=1e-100):
    """
    Find the minimum DKL between a point along  S0 - S1 - T <-> S1 - S0 - T and
    the given distribution.

    The points are the theta-geodesic q_lam ~ p(s0, s1) p(y | s0)^lam p(y | s1)^(1 - lam), with y the
    target, for any real lam (Niu & Quinn note that the optimal lam need not lie in [0, 1]). The
    divergence is convex in lam, since its lam-dependence is a log-partition function, so a scalar
    search finds the minimum. It is evaluated in the log domain, which keeps large |lam| finite.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the minimal DKL from.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.
    fuzz : float
        The amount to perturb the distribution by so that structural zeros don't
        impact the optimization.

    Returns
    -------
    s : float
        The minimum DKL.

    Raises
    ------
    ValueError
        Raised if the incorrect number of sources is supplied.
    """
    if len(sources) != 2:
        msg = "I_IG is a bivariate PID measure, and so requires exactly 2 sources."
        raise ValueError(msg)

    d = dist.coalesce(tuple(sources) + (target,))
    d.make_dense()
    p = d.pmf.reshape([len(a) for a in d.alphabet])

    if fuzz:
        p = p + fuzz
        p /= p.sum()

    log_p = np.log(p)
    log_s0s1 = np.log(p.sum(axis=2, keepdims=True))
    log_t_s0 = np.log(p.sum(axis=1, keepdims=True)) - np.log(p.sum(axis=(1, 2), keepdims=True))
    log_t_s1 = np.log(p.sum(axis=0, keepdims=True)) - np.log(p.sum(axis=(0, 2), keepdims=True))
    difference = np.broadcast_to(log_t_s0 - log_t_s1, p.shape)

    def objective(lam):
        log_q = log_s0s1 + log_t_s1 + lam * difference
        log_q = log_q - logsumexp(log_q)
        return float(np.sum(p * (log_p - log_q)) / np.log(2))

    if np.ptp(difference) < 1e-12:
        # the sources give the target the same channel, so the geodesic is a single point
        return objective(0.0)

    # The fuzz makes the objective nearly flat on wide intervals, so a local search from
    # anywhere could stall: bracket the minimum of the convex objective on a grid first.
    grid = list(np.linspace(-1.0, 2.0, 13))
    values = [objective(lam) for lam in grid]
    while np.argmin(values) in (0, len(grid) - 1):
        if np.argmin(values) == 0:
            grid.insert(0, grid[0] - 2 * (grid[1] - grid[0]))
            values.insert(0, objective(grid[0]))
        else:
            grid.append(grid[-1] + 2 * (grid[-1] - grid[-2]))
            values.append(objective(grid[-1]))
        if grid[-1] - grid[0] > 1e6:  # pragma: no cover
            break
    best = int(np.argmin(values))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    res = minimize_scalar(objective, bounds=(low, high), method="bounded", options={"xatol": 1e-12})

    return min(objective(res.x), values[best])


class PID_IG(BaseBivariatePID):
    """
    The Niu & Quinn partial information decomposition.
    """

    _name = "I_IG"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute the minimal DKL.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_ig for.
        sources : iterable of iterables, len(sources) == 2
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        red : float
            The redundancy value.
        """
        if len(sources) != 2:  # pragma: no cover
            msg = f"This method needs exact two sources, {len(sources)} given."
            raise ditException(msg)

        syn = ig_synergy(d, sources, target)

        co_i = coinformation(d, sources + (target,))

        red = co_i + syn

        return red
