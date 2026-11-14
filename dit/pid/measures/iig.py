"""
I_IG, Niu & Quinn.

This measure is inspired by information geometry, and finds a point between
S0 - S1 - T and S1 - S0 - T such that the DKL between that point and the true
distribution is minimized. That minimal DKL is then the synergy.
"""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, softmax

from ...exceptions import ditException
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_IG",)

# A one-dimensional geodesic is followed at most this far. If the divergence decreases
# without end along it, the value there equals its infimum to within rounding.
_CAP = 1e6


def _dense(dist, rvs, fuzz=0.0):
    """
    The joint distribution of `rvs` (a list of groups of variables) as a dense array,
    one axis per group.
    """
    d = dist.coalesce(rvs)
    d.make_dense()
    p = d.pmf.reshape([len(a) for a in d.alphabet])
    if fuzz:
        p = p + fuzz
        p /= p.sum()
    return p


def _log_marginal(p, axes):
    """
    The log of the marginal of `p` on `axes`, broadcastable to `p`, -inf where it is 0.
    """
    others = tuple(i for i in range(p.ndim) if i not in axes)
    with np.errstate(divide="ignore"):
        return np.log(p.sum(axis=others, keepdims=True))


def _log_projection(p, first, second):
    """
    The log of :math:`p(a)\\, p(b) / p(a \\cap b)` for the axes `first` and `second`: the
    projection of `p` onto the distributions in which the axes only in `first` are
    independent of those only in `second`, given the axes in both. It is -inf where the
    projection vanishes.
    """
    log_shared = _log_marginal(p, set(first) & set(second))
    with np.errstate(invalid="ignore"):
        log_q = _log_marginal(p, first) + _log_marginal(p, second) - log_shared
    return np.where(np.isfinite(log_shared), log_q, -np.inf)


def _argmin_convex(derivative, lo, hi):
    """
    A minimizer on [lo, hi] of a convex function, given its (nondecreasing) derivative.
    """
    if derivative(lo) >= 0:
        return lo
    if derivative(hi) <= 0:
        return hi
    return brentq(derivative, lo, hi, xtol=1e-15, maxiter=1000)


def _geodesic_divergence(p, log_projections):
    """
    The smallest KL divergence from `p` to the e-affine hull of some distributions.

    The hull is the family :math:`q_w \\propto \\prod_k P_k^{w_k}` with real weights
    :math:`w_k` that sum to 1. Where the :math:`P_k` have different supports, a :math:`P_k`
    that vanishes at an outcome where another is positive must have a nonnegative weight,
    or :math:`q_w` is infinite there; these are the weights for which every factor is
    finite, whatever positive values the zeros are replaced by. The minimum is taken over
    them, with :math:`q_w` restricted to the outcomes where every :math:`P_k` is positive.
    For two distributions this is the limit of adding a vanishing constant to every
    probability of `p`. The divergence is convex in the weights, since it depends on them
    through a log-partition function.

    Parameters
    ----------
    p : np.ndarray
        The distribution, as a dense array of probabilities.
    log_projections : list of np.ndarray
        The logs of the :math:`P_k`, broadcastable to the shape of `p`, -inf where they
        vanish. Every :math:`P_k` must be positive on the support of `p`.

    Returns
    -------
    divergence : float
        The minimal divergence, in bits.
    """
    logs = np.array([np.broadcast_to(log_q, p.shape).ravel() for log_q in log_projections])
    p = p.ravel()
    finite = np.isfinite(logs)
    common = finite.all(axis=0)
    # The projections that vanish somewhere another is positive need nonnegative weights.
    # With w_0 = 1 - sum(u) and w_k = u_k, these are the rows of A @ u + b >= 0.
    restricted = [not row.all() for row in finite[:, finite.any(axis=0)]]
    A = np.vstack([-np.ones(len(logs) - 1), np.eye(len(logs) - 1)])[restricted]
    b = np.eye(len(logs))[0][restricted]

    base, deltas = logs[0, common], logs[1:, common] - logs[0, common]
    p = p[common]
    support = p > 0
    constant = np.sum(p[support] * (np.log(p[support]) - base[support]))
    moments = deltas[:, support] @ p[support]

    def objective(u):
        return constant - u @ moments + logsumexp(base + u @ deltas)

    def gradient(u):
        return deltas @ softmax(base + u @ deltas) - moments

    def hessian(u):
        q = softmax(base + u @ deltas)
        mean = deltas @ q
        return (deltas * q) @ deltas.T - np.outer(mean, mean)

    if len(deltas) == 1:
        # u is the weight of the second distribution.
        lo = 0.0 if restricted[1] else -_CAP
        hi = 1.0 if restricted[0] else _CAP
        best = _argmin_convex(lambda x: gradient(np.array([x]))[0], lo, hi)
        value = objective(np.array([best]))
    else:
        # The barycenter satisfies every constraint strictly. SLSQP is only used when the
        # unconstrained minimizer leaves the feasible set: on flat objectives it can stop
        # while the divergence is still about 1e-10 above its minimum.
        start = np.full(len(deltas), 1 / len(logs))
        options = {"gtol": 1e-13, "maxiter": 1000}
        result = minimize(objective, start, jac=gradient, hess=hessian, method="trust-exact", options=options)
        if len(A) and np.min(A @ result.x + b) < -1e-9:
            constraint = {"type": "ineq", "fun": lambda u: A @ u + b, "jac": lambda u: A}
            options = {"ftol": 1e-15, "maxiter": 1000}
            result = minimize(objective, start, jac=gradient, method="SLSQP", constraints=[constraint], options=options)
        value = min(result.fun, objective(start))

    return value / np.log(2)


def ig_synergy(dist, sources, target, fuzz=0.0):
    """
    Find the minimum DKL between a point along  S0 - S1 - T <-> S1 - S0 - T and
    the given distribution.

    The points are the theta-geodesic q_lam ~ p(s0, s1) p(y | s0)^lam p(y | s1)^(1 - lam), with y the
    target, for any real lam (Niu & Quinn note that the optimal lam need not lie in [0, 1]). The
    divergence is convex in lam, since its lam-dependence is a log-partition function.

    Where the two Markov-chain distributions have different supports, q_lam is finite only for lam in
    [0, 1] (or a half-line), and is then supported on their common support. The minimum is taken over
    that interval: this is the limit of adding a vanishing constant to every probability.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the minimal DKL from.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.
    fuzz : float
        If positive, add this to every probability (and renormalize) first, which gives
        every outcome positive probability. The default, 0, computes the limit of this as
        the fuzz tends to 0 exactly.

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

    p = _dense(dist, [*sources, target], fuzz)
    return _geodesic_divergence(p, [_log_projection(p, (0, 1), (0, 2)), _log_projection(p, (0, 1), (1, 2))])


def ig_double_synergy(dist, sources, targets):
    """
    The information-geometric synergy between two sources and two targets: the
    synergistic-to-synergistic atom of PhiID-IG :cite:`liardi2026information`.

    It is the smallest KL divergence from the joint distribution to the e-affine hull of
    the four projections :math:`p(x_0, x_1, y_j)\\, p(x_i, y_0, y_1) / p(x_i, y_j)`, onto
    the distributions in which :math:`X_{1-i}` and :math:`Y_{1-j}` are independent given
    :math:`X_i` and :math:`Y_j`. Each projection removes the dependence between one source
    and one target alone, so the divergence is the dependence that needs all four
    variables. Where the supports of the projections differ, one that vanishes at an
    outcome where another is positive must have a nonnegative weight.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables
        The two source variables.
    targets : iterable of iterables
        The two target variables.

    Returns
    -------
    s : float
        The synergy, in bits.

    Raises
    ------
    ValueError
        Raised unless there are exactly two sources and two targets.
    """
    if len(sources) != 2 or len(targets) != 2:
        msg = "The double synergy requires exactly 2 sources and 2 targets."
        raise ValueError(msg)

    p = _dense(dist, [*sources, *targets])
    projections = [_log_projection(p, (0, 1, j), (i, 2, 3)) for i in (0, 1) for j in (2, 3)]
    return _geodesic_divergence(p, projections)


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
