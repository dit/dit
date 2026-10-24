"""
The Blahut-Arimoto algorithm for solving the rate-distortion problem.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..algorithms.optimization import _resolve_n_jobs
from ..divergences.pmf import relative_entropy
from ..math.sampling import sample_simplex
from .distortions import hamming_distortion
from .rate_distortion import RateDistortionResult

__all__ = (
    "blahut_arimoto",
    "blahut_arimoto_ib",
)


###############################################################################
# Rate-Distortion


def _blahut_arimoto(p_x, beta, q_y_x, distortion, max_iters=10_000, tol=1e-12):
    """
    Perform the Blahut-Arimoto algorithm.

    Parameters
    ----------
    p_x : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    distortion : np.ndarray
        The distortion matrix.
    max_iters : int
        The maximum number of iterations.
    tol : float
        For a ``q_independent`` distortion, stop once Blahut's bounds on the
        Lagrangian agree to within ``tol`` bits.

    Returns
    -------
    result : RateDistortionResult
        A rate, distortion pair.
    q_xy : np.ndarray
        The joint distribution q(x, y).
    """

    def q_xy(q_y_x):
        """
        :math:`q(x,y) = q(y|x)p(x)`
        """
        q_xy = p_x[:, np.newaxis] * q_y_x
        return q_xy

    def next_q_y(q_y_x):
        """
        :math:`q(y) = \\sum_x q(y|x)p(x)`
        """
        q_y = np.matmul(p_x, q_y_x)
        return q_y

    def tilt(d):
        """
        :math:`2^{-\\beta d(x, y)}`, rescaled per row so the largest entry is 1.
        """
        bd = beta * d
        finite = np.isfinite(bd)
        row_min = np.min(np.where(finite, bd, np.inf), axis=1, keepdims=True)
        row_min = np.where(np.isfinite(row_min), row_min, 0.0)
        return np.exp2(-(bd - row_min))

    def next_q_y_x(q_y, d):
        """
        :math:`q(y|x) = q(y) 2^{-\\beta * distortion}`
        """
        q_y_x = q_y * tilt(d)
        q_y_x /= q_y_x.sum(axis=1, keepdims=True)
        return q_y_x

    def bound_gap(q_y, tilted):
        """
        Gap between Blahut's upper and lower bounds on
        :math:`\\min I + \\beta \\langle d \\rangle` :cite:`Blahut1972`:
        :math:`\\log_2 \\max_y \\sum_x p(x) 2^{-\\beta d(x, y)} / c(x)` with
        :math:`c(x) = \\sum_y q(y) 2^{-\\beta d(x, y)}`.
        """
        c = tilted @ q_y
        return float(np.log2(np.max((p_x / c) @ tilted)))

    def av_dist(q_y_x, dist):
        """
        :math:`<dist> = \\sum_{x, t} q(x,t) * d(x,t)`
        """
        # ``q_y_x`` can contain zeros while ``dist`` contains infs (e.g. residual
        # entropy distortion), giving ``0 * inf = nan`` intermediates.
        with np.errstate(invalid="ignore"):
            d = np.matmul(p_x, (q_y_x * dist)).sum()
        return d

    fixed = getattr(distortion, "q_independent", False)
    dist = distortion(p_x, q_y_x)
    tilted = tilt(dist) if fixed else None

    # A q-independent distortion makes the problem convex, so Blahut's bounds
    # certify convergence. Otherwise stop when the channel itself stops moving:
    # D alone can stall near the critical slope long before the Lagrangian has
    # converged.
    for _ in range(max_iters):
        q_y = next_q_y(q_y_x)
        if fixed:
            if bound_gap(q_y, tilted) < tol:
                break
            q_y_x = q_y * tilted
            q_y_x /= q_y_x.sum(axis=1, keepdims=True)
            continue
        prev_q_y_x = q_y_x
        q_y_x = next_q_y_x(q_y, dist)
        dist = distortion(p_x, q_y_x)
        if np.max(np.abs(q_y_x - prev_q_y_x)) < 1e-12:
            break

    d = av_dist(q_y_x, dist)
    q = q_xy(q_y_x)
    # Zeros in ``q`` give ``log2(0)``/``0/0`` intermediates handled by ``nansum``.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.nansum(q * np.log2(q / (q.sum(axis=0, keepdims=True) * q.sum(axis=1, keepdims=True))))
    result = RateDistortionResult(r, d)

    return result, q


def blahut_arimoto(p_x, beta, distortion=hamming_distortion, max_iters=10_000, restarts=100):
    """
    Perform a robust form of the Blahut-Arimoto algorithms.

    Parameters
    ----------
    p_x : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    distortion : np.ndarray
        The distortion matrix.
    max_iters : int
        The maximum number of iterations.
    restarts : int
        The number of initial conditions to try. A distortion marked
        ``q_independent`` makes the problem convex, so only the uniform channel
        and the best constant channel are tried.

    Returns
    -------
    result : RateDistortionResult
        The rate, distortion pair.
    q_xy : np.ndarray
        The distribution p(x, y) which achieves the optimal rate, distortion.
    """
    n = len(p_x)
    constant = 0
    if getattr(distortion, "q_independent", False):
        restarts = min(restarts, 2)
        uniform = np.ones((n, n)) / n
        constant = int(np.argmin(p_x @ distortion(p_x, uniform)))

    # Pre-generate all initial conditions serially so the global RNG state is
    # consumed deterministically regardless of whether the subsequent (pure,
    # RNG-free) ``_blahut_arimoto`` calls run serially or across threads.
    inits = []
    for i in range(restarts):
        if i == 0:
            q_y_x = np.ones((n, n)) / n
        elif i == 1:
            # the constant channel: every x maps to the same reconstruction
            q_y_x = np.zeros((n, n))
            q_y_x[:, constant] = 1
        else:
            q_y_x = sample_simplex(n, n)
        inits.append(q_y_x)

    def _run(q_y_x):
        return _blahut_arimoto(p_x=p_x, beta=beta, q_y_x=q_y_x, distortion=distortion, max_iters=max_iters)

    n_jobs = _resolve_n_jobs(len(inits))
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            candidates = list(executor.map(_run, inits))
    else:
        candidates = [_run(q_y_x) for q_y_x in inits]

    rd = min(candidates, key=lambda result: result[0].rate + beta * result[0].distortion)
    return rd


###############################################################################
# Information Bottleneck


def blahut_arimoto_ib(p_xy, beta, divergence=relative_entropy, max_iters=100, restarts=250):  # pragma: no cover
    """
    Perform a robust form of the Blahut-Arimoto algorithms.

    Parameters
    ----------
    p_xy : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    divergence : func
        The divergence measure to construct a distortion from: D(p(Y|x)||q(Y|t)).
    max_iters : int
        The maximum number of iterations.
    restarts : int
        The number of initial conditions to try.

    Returns
    -------
    result : RateDistortionResult
        The rate, distortion pair.
    q_xyt : np.ndarray
        The distribution p(x, y, t) which achieves the optimal rate, distortion.
    """
    p_x = p_xy.sum(axis=1)
    p_y_x = p_xy / p_xy.sum(axis=1, keepdims=True)

    def next_q_y_t(q_t_x):
        """
        :math:`q(y|t) = (\\sum_x p(x, y) * q(t|x)) / q(t)`
        """
        q_xyt = q_t_x[:, np.newaxis, :] * p_xy[:, :, np.newaxis]
        q_ty = q_xyt.sum(axis=0).T
        q_y_t = q_ty / q_ty.sum(axis=1, keepdims=True)
        q_y_t[np.isnan(q_y_t)] = 1
        return q_y_t

    def distortion(p_x, q_t_x):
        """
        :math:`d(x, t) = D[ p(Y|x) || q(Y|t) ]`
        """
        q_y_t = next_q_y_t(q_t_x)
        distortions = np.asarray([divergence(a, b) for a in p_y_x for b in q_y_t]).reshape(len(p_y_x), len(q_y_t))
        return distortions

    rd, q_xt = blahut_arimoto(p_x=p_x, beta=beta, distortion=distortion, max_iters=max_iters, restarts=restarts)

    sums = q_xt.sum(axis=1, keepdims=True)
    q_t_x = q_xt / sums
    q_t_x[np.isnan(q_t_x)] = 1 / q_t_x.shape[1]

    q_xyt = p_xy[:, :, np.newaxis] * q_t_x[:, np.newaxis, :]

    return rd, q_xyt


###############################################################################
# TODO: Deterministic Forms
