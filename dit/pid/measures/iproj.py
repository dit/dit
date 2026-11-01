"""
The I_proj measure as proposed by Harder et al.
"""

import numpy as np
from scipy.optimize import minimize

from ...exceptions import ditException
from ..pid import BaseBivariatePID

__all__ = ("PID_Proj",)


def _reverse_projection(p, domain, tol=1e-12):
    """
    The distribution in the convex hull of the rows of `domain` closest to `p`:
    :math:`\\arg\\min_{q \\in C_{cl}(\\text{domain})} D(p \\| q)`.

    The divergence is convex in the weights w of q = w @ domain, with gradient -c, where
    c_j = sum_z p(z) domain[j, z] / q(z). It is minimized by SLSQP from a few steps of the
    EM fixed point w_j <- w_j c_j, and then polished by that fixed point, which never
    increases it. Since sum_j w_j c_j = 1, convexity bounds the excess divergence by
    max_j c_j - 1. The EM fixed point alone converges slowly when the minimum lies on a
    face of the hull that some other row touches tangentially.

    Parameters
    ----------
    p : np.ndarray
        The distribution to project.
    domain : np.ndarray
        The distributions spanning the convex hull, one per row.
    tol : float
        The bound on the excess divergence, in nats, at which to stop polishing.

    Returns
    -------
    q : np.ndarray
        The projection of `p`.
    """
    support = p > 0
    p_s, domain_s = p[support], domain[:, support]

    def divergence(w):
        return -np.sum(p_s * np.log(w @ domain_s))

    def gradient(w):
        return -(domain_s @ (p_s / (w @ domain_s)))

    def em(w, steps):
        for _ in range(steps):
            ratios = domain_s @ (p_s / (w @ domain_s))
            if ratios.max() - 1 <= tol:
                break
            w = w * ratios
            w /= w.sum()
        return w

    start = em(np.full(len(domain), 1 / len(domain)), 20)
    res = minimize(
        divergence,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * len(domain),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-16, "maxiter": 500},
    )
    weights = np.clip(res.x, 0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else start
    if not np.isfinite(divergence(weights)) or divergence(weights) > divergence(start):
        weights = start
    weights = em(weights, 2000)
    return weights @ domain


def projected_information(dist, X, Y, Z):
    """
    :math:`I_Z^\\pi(X \\searrow Y)`

    Paramters
    ---------
    dist : Distribution
        The distribution to compute the projected information from.
    X : iterable
        The aggregate variable X.
    Y : iterable
        The aggregate variable Y.
    Z : iterable
        The aggregate variable Z.

    Returns
    -------
    pi : float
        The projected information.
    """
    d = dist.coalesce([X, Y, Z])
    d.make_dense()
    p = np.asarray(d.pmf, dtype=float).reshape([len(a) for a in d.alphabet])
    p_xz, p_yz = p.sum(axis=1), p.sum(axis=0)
    p_x, p_y, p_z = p_xz.sum(axis=1), p_yz.sum(axis=1), p_xz.sum(axis=0)
    domain = p_yz[p_y > 0] / p_y[p_y > 0, None]

    value = 0.0
    for p_xz_x, p_x_x in zip(p_xz, p_x, strict=True):
        if p_x_x > 0:
            projection = _reverse_projection(p_xz_x / p_x_x, domain)
            mask = p_xz_x > 0
            value += np.sum(p_xz_x[mask] * np.log2(projection[mask] / p_z[mask]))
    return float(value)


class PID_Proj(BaseBivariatePID):
    """
    The Harder et al partial information decomposition.
    """

    _name = "I_↘"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute

        .. math::
            I_proj(sources : target) = min{PI(X \\searrow Y), PI(Y \\searrow X)}

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_proj for.
        sources : iterable of iterables, len(sources) == 2
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        iproj : float
            The value of I_proj.
        """
        if len(sources) != 2:  # pragma: no cover
            msg = f"This method needs exact two sources, {len(sources)} given."
            raise ditException(msg)

        pi_0 = projected_information(d, sources[0], sources[1], target)
        pi_1 = projected_information(d, sources[1], sources[0], target)
        return min(pi_0, pi_1)
