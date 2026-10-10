"""
Compute the hypercontractivity coefficient:
    s*(X||Y) = max_{U - X - Y} I[U:Y]/I[U:X]
"""

from itertools import product
from math import comb

import numpy as np
from scipy.optimize import linprog

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..multivariate.entropy import entropy
from ..multivariate.total_correlation import total_correlation

__all__ = (
    "HypercontractivityCoefficient",
    "hypercontractivity_coefficient",
)


def _max_deterministic_aux(hc, max_channels=4096):
    """
    Maximize the hypercontractivity ratio over deterministic auxiliary channels.

    Basin hopping often stalls at the independent-``U`` point (ratio ``0``); when
    the auxiliary alphabet is small, an exhaustive sweep over one-hot channels is
    cheap and reliably finds better witnesses.
    """
    if len(hc._aux_vars) != 1:
        return None

    shape = tuple(hc._aux_vars[0].shape)
    n_out = shape[-1]
    in_shape = shape[:-1]
    n_in = int(np.prod(in_shape)) if in_shape else 1
    if n_out**n_in > max_channels:
        return None

    best = -np.inf
    for choices in product(range(n_out), repeat=n_in):
        channel = np.zeros(shape)
        for idx, out in zip(np.ndindex(in_shape), choices, strict=True):
            channel[idx + (out,)] = 1.0
        ratio = -hc.objective(channel.reshape(-1))
        if np.isfinite(ratio):
            best = max(best, ratio)

    return best if np.isfinite(best) else None


def _markov_witness_ratios(dist, rv_x, rv_y):
    """
    Lower bounds from variables already in ``dist`` that satisfy W - X - Y.
    """
    rv_x = tuple(rv_x)
    rv_y = tuple(rv_y)
    used = set(rv_x) | set(rv_y)
    witnesses = [w for w in range(dist.outcome_length()) if w not in used]
    if not witnesses:
        return None

    tiny = np.finfo(float).tiny
    best = 0.0
    cond = list(rv_x)
    for w in witnesses:
        if not np.isclose(total_correlation(dist, [[w], rv_y], cond), 0.0):
            continue
        denom = total_correlation(dist, [[w], rv_x])
        if denom <= tiny:
            continue
        numer = total_correlation(dist, [[w], rv_y])
        best = max(best, numer / denom)

    return best if best > 0 else None


def _joint_matrix(dist, rv_x, rv_y):
    """The joint pmf of (X, Y) as a matrix, restricted to the support of X."""
    d = dist.coalesce([list(rv_x), list(rv_y)])
    d.make_dense()
    pmf = d.pmf.reshape([len(a) for a in d.alphabet])
    return pmf[pmf.sum(axis=1) > 0]


def _maximal_correlation_squared(pmf):
    """
    Squared maximal correlation, the second singular value of p(x,y)/sqrt(p(x)p(y)).
    It lower-bounds s*, approached as U becomes independent of X :cite:`anantharam2013maximal`.
    """
    px, py = pmf.sum(axis=1), pmf.sum(axis=0)
    keep = py > 0
    B = pmf[:, keep] / np.sqrt(np.outer(px, py[keep]))
    sv = np.linalg.svd(B, compute_uv=False)
    return float(sv[1] ** 2) if len(sv) > 1 else 0.0


def _envelope_hypercontractivity(pmf, max_alphabet=4, grid_points=3000, iters=30, tol=1e-9):
    """
    Lower bound on s* from the convex-envelope characterization
    :cite:`anantharam2013maximal`: with ``g(q) = H(q W) - lam H(q)`` for X ~ q,
    ``I(U:Y) - lam I(U:X) = g(p) - sum_u p(u) g(q_u)``, so s* is the least
    ``lam`` at which ``g`` touches its convex envelope at ``p(x)``. Envelopes are
    taken over a grid of Alice's simplex; the returned ``lam`` is violated by an
    explicit split, so it never exceeds s*.
    """
    n = pmf.shape[0]
    if n > max_alphabet:
        return None
    px = pmf.sum(axis=1)
    W = pmf / px[:, None]
    count = 1
    while comb(count + n, n - 1) <= grid_points:
        count += 1
    grid = [c for c in product(range(count + 1), repeat=n - 1) if sum(c) <= count]
    Q = np.array([list(c) + [count - sum(c)] for c in grid], dtype=float) / count
    Q = np.r_[Q, px[None, :]]

    def ent(P):
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.where(P > 0, P * np.log2(P), 0.0).sum(axis=-1)

    hy, hx = ent(Q @ W), ent(Q)

    def violated(lam):
        g = hy - lam * hx
        res = linprog(g, A_eq=Q.T, b_eq=px, bounds=(0, None), method="highs")
        return res.x is not None and g[-1] - res.fun > tol

    lo, hi = 0.0, 1.0
    if not violated(lo):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if violated(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _product_hypercontractivity(dist, rv_x, rv_y, bound, niter):
    """
    Exact tensorization when ``dist`` factors into independent blocks.

    For ``dist = dist_left ⊗ dist_right`` with
    ``(X, Y) = (X_left X_right, Y_left Y_right)``,
    ``s*(X:Y) = max(s*(X_left:Y_left), s*(X_right:Y_right))``.
    """
    rv_x = list(rv_x)
    rv_y = list(rv_y)
    if len(rv_x) < 2 or len(rv_y) < 2 or len(rv_x) != len(rv_y):
        return None

    mid = len(rv_x) // 2
    left = sorted(rv_x[:mid] + rv_y[:mid])
    right = sorted(rv_x[mid:] + rv_y[mid:])
    if not np.isclose(total_correlation(dist, [left, right]), 0.0):
        return None

    k = mid
    pairs = [list(range(k)), list(range(k, 2 * k))]
    hc_l = hypercontractivity_coefficient(dist.marginal(left), pairs, bound=bound, niter=niter)
    hc_r = hypercontractivity_coefficient(dist.marginal(right), pairs, bound=bound, niter=niter)
    return max(hc_l, hc_r)


class HypercontractivityCoefficient(BaseAuxVarOptimizer):
    """
    Computes the hypercontractivity coefficient:

    .. math::
        max_{U - X - Y} I[U:Y] / I[U:X]
    """

    _shotgun = 5

    def __init__(self, dist, rv_x=None, rv_y=None, bound=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rv_x : iterable
            The variables to consider `X`.
        rv_y : iterable
            The variables to consider `Y`.
        bound : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        """
        self._x = {0}
        self._y = {1}
        self._u = {3}
        super().__init__(dist, [rv_x, rv_y], [])

        theoretical_bound = self._full_shape[self._proxy_vars[0]] + 1
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([({0}, bound)])

    def _objective(self):
        """
        The hypercontractivity coefficient to minimize.

        Returns
        -------
        obj : func
            The objective function.
        """
        mi_a = self._mutual_information(self._u, self._y)
        mi_b = self._mutual_information(self._u, self._x)

        def objective(self, x):
            """
            Compute :math:`I[U:Y] / I[U:X]`

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            pmf = self.construct_joint(x)
            a = mi_a(pmf)
            b = mi_b(pmf)
            # ``U - X - Y`` is a Markov chain, so the data-processing inequality
            # guarantees ``0 <= I[U:Y] <= I[U:X]`` and hence the ratio lies in
            # ``[0, 1]``. As ``I[U:X] -> 0`` the ratio is a numerically unstable
            # ``0/0``: a denominator of, say, ``1e-8`` turns estimator noise in
            # the numerator into a spurious ratio far above 1, and basin-hopping
            # is happy to chase those artifacts. Guard the denominator well above
            # machine ``tiny`` and clamp to the theoretical ``[0, 1]`` range so
            # the optimizer cannot be lured to degenerate, near-independent ``U``.
            eps = 1e-6
            if b <= eps:
                return 0.0
            return -min(1.0, max(0.0, a / b))

        return objective


def hypercontractivity_coefficient(dist, rvs, bound=None, niter=None):
    """
    Computes the hypercontractivity coefficient:

    .. math::
        s*(X||Y) = max_{U - X - Y} I[U:Y]/I[U:X]

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The variables to compute the hypercontractivity coefficient of.
        Order is important.
    bound : int, None
        An external bound on the size of `U`. If None, :math:`|U| <= |X|+1`.
    niter : int, None
        The number of basin-hopping steps to perform. If None, use the default.

    Returns
    -------
    hc : float
        The hypercontractivity coefficient.
    """
    rvs, _ = normalize_rvs(dist, rvs, None)

    if len(rvs) != 2:
        msg = f"Hypercontractivity coefficient can only be computed for 2 variables, not {len(rvs)}."
        raise ditException(msg)

    # test some special cases:
    if np.isclose(total_correlation(dist, rvs), 0.0):
        return 0.0
    elif np.isclose(entropy(dist, rvs[1], rvs[0]), 0.0):
        return 1.0

    product = _product_hypercontractivity(dist, rvs[0], rvs[1], bound, niter)
    if product is not None:
        return float(product)

    hc = HypercontractivityCoefficient(dist, rvs[0], rvs[1], bound=bound)
    hc.optimize(niter=niter)
    val = -hc.objective(hc._optima)
    det = _max_deterministic_aux(hc)
    if det is not None:
        val = max(val, det)
    wit = _markov_witness_ratios(dist, rvs[0], rvs[1])
    if wit is not None:
        val = max(val, wit)
    pmf = _joint_matrix(dist, rvs[0], rvs[1])
    val = max(val, _maximal_correlation_squared(pmf))
    env = _envelope_hypercontractivity(pmf)
    if env is not None:
        val = max(val, env)
    if not np.isfinite(val):
        return np.inf if np.isneginf(val) else 0.0
    return float(max(0.0, val))
