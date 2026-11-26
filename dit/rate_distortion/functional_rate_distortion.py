"""
The functional rate-distortion function.

The functional rate-distortion function :cite:`flamich2026functional` is the
rate-distortion function with the mutual information replaced by the functional
information :math:`I_F(X \\to Y)`. It characterizes one-shot, variable-length
lossy compression to within :math:`\\log_2 e + 1` bits.
"""

import numpy as np
from scipy.optimize import linprog

from ..exceptions import OptimizationException, ditException
from ..other.functional_information import functional_information_pmf
from .distortions import Distortion, hamming_distortion
from .rate_distortion import RateDistortionResult

__all__ = ("functional_rate_distortion",)


def _distortion_matrix(p_x, distortion):
    """The distortion as a nonnegative ``(|X|, |Y|)`` matrix, Hamming by default."""
    n = len(p_x)
    if distortion is None:
        distortion = hamming_distortion
    if isinstance(distortion, Distortion):
        distortion = distortion.matrix
    if callable(distortion) and getattr(distortion, "q_independent", False):
        distortion = distortion(p_x, np.full((n, n), 1 / n))
    if distortion is None or callable(distortion):
        msg = "The distortion must not depend on the channel."
        raise ditException(msg)
    matrix = np.asarray(distortion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != n or not np.all(matrix >= 0):
        msg = "The distortion must be a nonnegative matrix with one row per outcome of X."
        raise ditException(msg)
    return matrix


def _greedy_vertex(column, p_x):
    """
    The vertex of the base polytope of :math:`F(A) = \\eta(P_X(A))` chosen by
    the greedy algorithm for ``column``.

    Its inner product with ``column`` is the Lovász extension of :math:`F` there
    (the column's contribution to :math:`I_F`), and with any other vector it is
    at most the extension's value: :math:`F` is submodular, so the extension is
    the maximum over these vertices.
    """
    order = np.argsort(-column, kind="stable")
    levels = np.cumsum(p_x[order])
    eta = -levels * np.log2(levels)
    vertex = np.empty_like(p_x)
    vertex[order] = np.diff(eta, prepend=0.0)
    return vertex


def functional_rate_distortion(p_x, D, distortion=None, tol=1e-9, maxiter=10_000):
    """
    The functional rate-distortion function :cite:`flamich2026functional`
    (Definition 8),

    .. math::

        R_F(D) = \\min_{p(y|x)\\,:\\,\\mathbb{E}[\\Delta(X, Y)] \\leq D} I_F(X \\to Y),

    where :math:`I_F` is the functional information
    (:func:`dit.other.functional_information`). It brackets the least entropy
    of a reproduction meeting the distortion constraint, when the encoder may
    use private randomness (Theorem 2):

    .. math::

        R_F(D) \\leq R_O(D) \\leq R_F(D) + \\log_2 e + 1.

    Since :math:`I_F \\geq I`, it lies above the rate-distortion function. It
    is nonincreasing, convex, and piecewise linear in :math:`D` (Proposition 9).
    For a Bernoulli(:math:`p`) source under Hamming distortion it is
    :math:`h(p) (1 - D / \\min(p, 1 - p))` below :math:`\\min(p, 1 - p)`
    (eq. 70).

    Sorting the column :math:`k_y = p(y|\\cdot)` of the channel gives
    :math:`I_F = \\sum_y f(k_y)`, with :math:`f` the Lovász extension of the
    submodular set function :math:`F(A) = \\eta(P_X(A))`,
    :math:`\\eta(s) = -s \\log_2 s` (Lemma 3). So :math:`f(k_y)` is the largest
    of the linear functions given by the vertices of the base polytope of
    :math:`F`, and the minimization is a linear program over the channel and an
    epigraph variable per output. It is solved by cutting planes: sorting a
    column finds its most violated cut exactly, and there are finitely many,
    so the value is exact.

    Parameters
    ----------
    p_x : array_like, shape (n_x,)
        The source pmf.
    D : float
        The largest allowed expected distortion.
    distortion : array_like, Distortion, callable, None
        The distortion :math:`\\Delta(x, y)` as an ``(n_x, n_y)`` matrix, whose
        entries may be ``numpy.inf`` to forbid a reproduction. A
        :class:`Distortion` or a distortion function marked ``q_independent``
        (such as ``hamming``) is converted to its matrix. Defaults to the
        Hamming distortion.
    tol : float
        A cut is added when it is violated by more than this.
    maxiter : int
        The largest number of linear programs to solve.

    Returns
    -------
    result : RateDistortionResult
        The rate :math:`R_F(D)`, as the functional information of the returned
        channel, and that channel's expected distortion (at most ``D``).
    q_xy : np.ndarray, shape (n_x, n_y)
        The joint distribution :math:`p(x) q(y|x)` of an optimal channel.

    Raises
    ------
    ditException
        If the distortion is not a fixed nonnegative matrix, or no channel has
        expected distortion at most ``D``.
    OptimizationException
        If a linear program fails or the cuts do not converge within
        ``maxiter`` programs.
    """
    p_x = np.asarray(p_x, dtype=float)
    delta = _distortion_matrix(p_x, distortion)
    support = p_x > 0
    px, allowed = p_x[support], np.isfinite(delta[support])
    cost = np.where(allowed, delta[support], 0.0)
    n, m = cost.shape
    least = px @ np.where(allowed, cost, np.inf).min(axis=1)
    if not least <= D:
        msg = f"No channel has expected distortion at most {D}; the least is {least}."
        raise ditException(msg)

    # Variables: the channel k[x, y] (flattened row-major), then t[y] >= f(k[:, y]).
    nk = n * m
    objective = np.r_[np.zeros(nk), np.ones(m)]
    a_eq = np.kron(np.eye(n), np.ones((1, m)))
    a_eq = np.hstack([a_eq, np.zeros((n, m))])
    rows = [np.r_[(px[:, np.newaxis] * cost).ravel(), np.zeros(m)]]
    bounds = [(0, 1 if ok else 0) for ok in allowed.ravel()] + [(0, None)] * m
    for _ in range(maxiter):
        res = linprog(
            objective,
            A_ub=np.array(rows),
            b_ub=np.r_[D, np.zeros(len(rows) - 1)],
            A_eq=a_eq,
            b_eq=np.ones(n),
            bounds=bounds,
            method="highs",
        )
        if res.status != 0:
            raise OptimizationException(res.message)
        k, t = res.x[:nk].reshape(n, m), res.x[nk:]
        violated = False
        for y in range(m):
            vertex = _greedy_vertex(k[:, y], px)
            if vertex @ k[:, y] > t[y] + tol:
                cut = np.zeros(nk + m)
                cut[np.arange(n) * m + y] = vertex
                cut[nk + y] = -1
                rows.append(cut)
                violated = True
        if not violated:
            break
    else:
        msg = f"The cutting planes did not converge within {maxiter} linear programs."
        raise OptimizationException(msg)

    k = np.clip(k, 0, None)
    k /= k.sum(axis=1, keepdims=True)
    q_xy = np.zeros((len(p_x), m))
    q_xy[support] = px[:, np.newaxis] * k
    rate = functional_information_pmf(q_xy)
    achieved = float((q_xy[support] * cost).sum())
    return RateDistortionResult(rate, achieved), q_xy
