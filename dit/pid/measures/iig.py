"""
I_IG, Niu & Quinn.

This measure is inspired by information geometry, and finds a point between
S0 - S1 - T and S1 - S0 - T such that the DKL between that point and the true
distribution is minimized. That minimal DKL is then the synergy.
"""

import numpy as np

from scipy.optimize import minimize

from ...exceptions import ditException
from ...multivariate import coinformation
from ..pid import BaseBivariatePID


__all__ = (
    'PID_IG',
)


def ig_synergy(dist, sources, target, fuzz=1e-100):
    """
    Find the minimum DKL between a point along  S0 - S1 - T <-> S1 - S0 - T and
    the given distribution.

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
    ditException
        Raised if the optimization fails.
    """
    if len(sources) != 2:
        msg = "I_IG is a bivariate PID measure, and so requires exactly 2 sources."
        raise ValueError(msg)

    d = dist.coalesce(tuple(sources) + (target,))
    d.make_dense()
    d = d.pmf.reshape([len(a) for a in d.alphabet])

    if fuzz:
        d += fuzz
        d /= d.sum()

    p_s0s1 = d.sum(axis=2, keepdims=True)
    p_s0 = d.sum(axis=(1, 2), keepdims=True)
    p_s1 = d.sum(axis=(0, 2), keepdims=True)

    p_t_s0 = d.sum(axis=1, keepdims=True) / p_s0
    p_t_s1 = d.sum(axis=0, keepdims=True) / p_s1

    def p_star(t):
        d = p_s0s1 * p_t_s0 ** t * p_t_s1 ** (1 - t)
        d /= d.sum()
        return d

    def objective(t):
        dkl = (d * np.log2(d / p_star(t))).sum().item()
        return dkl

    res = minimize(fun=objective,
                   x0=np.random.random(),
                   method='L-BFGS-B',
                   options={'maxiter': 1000,
                            'ftol': 1e-10,
                            'eps': 1.4901161193847656e-08,
                            },
                   )

    if not res.success:  # pragma: no cover
        msg = f"Optimization failed: {res.message}"
        raise ditException(msg)

    return objective(res.x)


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
            msg = "This method needs exact two sources, {} given.".format(len(sources))
            raise ditException(msg)

        syn = ig_synergy(d, sources, target)

        co_i = coinformation(d, sources + (target,))

        red = co_i + syn

        return red
