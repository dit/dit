"""
Alternating divergence minimization for BROJA unique information.

Implements the admUI algorithm of Banerjee, Rauh, and Montúfar
(:cite:`banerjee2017computing`, arXiv:1709.07487).
"""

from __future__ import annotations

import numpy as np

from ..distribution import Distribution
from .pid_broja import prepare_dist as broja_prepare_dist

__all__ = (
    "admui_dist",
    "admui_optimize",
)

_DEFAULT_MAXITER = 1000


def _iproj_is(px_given_s: np.ndarray, py_given_s: np.ndarray, r_xy: np.ndarray, *, eps: float, maxiter: int):
    """
  Iterative-scaling I-projection of ``r_xy`` onto the set of joints with
  marginals ``px_given_s``, ``py_given_s``.
  """
    n_x, n_y = r_xy.shape
    x_mask = px_given_s > 0
    y_mask = py_given_s > 0
    b = r_xy[np.ix_(x_mask, y_mask)].copy()

    for _ in range(maxiter):
        row_sums = b.sum(axis=1)
        factor_x = px_given_s[x_mask] / row_sums
        b *= factor_x[:, np.newaxis]

        col_sums = b.sum(axis=0)
        factor_y = py_given_s[y_mask] / col_sums
        b *= factor_y[np.newaxis, :]

        diff = float(np.max(factor_x) * np.max(factor_y))
        if diff < 1.0 + eps:
            break

    return b, x_mask, y_mask


def _iproj_gis(px_given_s: np.ndarray, py_given_s: np.ndarray, r_xy: np.ndarray, *, eps: float, maxiter: int):
    """Generalized iterative-scaling I-projection."""
    n_x, n_y = r_xy.shape
    x_mask = px_given_s > 0
    y_mask = py_given_s > 0
    b = r_xy[np.ix_(x_mask, y_mask)].copy()

    denom = np.sqrt(px_given_s[x_mask, np.newaxis] * py_given_s[np.newaxis, y_mask])

    for _ in range(maxiter):
        row_sums = b.sum(axis=1)
        col_sums = b.sum(axis=0)
        oosbx = np.sqrt(1.0 / row_sums)
        oosby = np.sqrt(1.0 / col_sums)
        factor = denom * oosbx[:, np.newaxis] * oosby[np.newaxis, :]
        b *= factor
        if float(np.max(factor)) < 1.0 + eps:
            break

    return b, x_mask, y_mask


def admui_optimize(
    px_given_s: np.ndarray,
    py_given_s: np.ndarray,
    ps: np.ndarray,
    *,
    eps: float = 1e-7,
    maxiter: int = _DEFAULT_MAXITER,
    ip_method: str = "IS",
):
    """
    Run the admUI outer loop.

    Parameters
    ----------
    px_given_s : ndarray, shape (n_x, n_s)
        Conditional P(X | S).
    py_given_s : ndarray, shape (n_y, n_s)
        Conditional P(Y | S).
    ps : ndarray, shape (n_s,) or (n_s, 1)
        Marginal P(S).
    eps : float
        Outer-loop tolerance.
    maxiter : int
        Maximum outer iterations.
    ip_method : {'IS', 'GIS'}
        Inner I-projection algorithm.

    Returns
    -------
    q_sxy : ndarray, shape (n_s, n_x, n_y)
        Optimal joint Q(S, X, Y).
    n_iters : int
        Number of outer iterations performed.
    converged : bool
        Whether the outer loop met the tolerance.
    """
    ps = np.asarray(ps, dtype=float).reshape(-1)
    n_x, n_s = px_given_s.shape
    n_y = py_given_s.shape[0]
    n_xy = n_x * n_y

    if ip_method == "IS":
        iproj = _iproj_is
    elif ip_method == "GIS":
        iproj = _iproj_gis
    else:
        msg = f"Unknown ip_method {ip_method!r}; expected 'IS' or 'GIS'."
        raise ValueError(msg)

    eps_inner = eps / (20 * n_s)

    px_marg = px_given_s @ ps
    py_marg = py_given_s @ ps
    r_xy = 1e-6 * np.ones((n_x, n_y)) / n_xy + (1 - 1e-6) * np.outer(px_marg, py_marg)

    q_xy_given_s = np.zeros((n_x, n_y, n_s))
    converged = False

    for it in range(maxiter):
        diff = 1.0
        for s in range(n_s):
            ip, x_mask, y_mask = iproj(px_given_s[:, s], py_given_s[:, s], r_xy, eps=eps_inner, maxiter=maxiter)
            mask = np.outer(x_mask, y_mask)
            ip_flat = ip.ravel()
            old = q_xy_given_s[:, :, s][mask]
            if old.size == 0 or np.any(old <= 0):
                diffs = 2.0
            else:
                diffs = float(np.max(ip_flat / old))
            if diffs > diff:
                diff = diffs
            q_xy_given_s[:, :, s][mask] = ip_flat

        r_xy = np.tensordot(q_xy_given_s, ps, axes=([2], [0]))

        if diff - 1.0 < eps:
            converged = True
            break

    q_sxy = (q_xy_given_s * ps[np.newaxis, np.newaxis, :]).transpose(2, 0, 1)
    return q_sxy, it + 1, converged


def _conditionals_from_dist(d: Distribution):
    """Extract P(X|S), P(Y|S), P(S) from coalesced (X, Y, target) distribution."""
    n_x = len(d.alphabet[0])
    n_y = len(d.alphabet[1])
    n_s = len(d.alphabet[2])
    # alphabet order: [0]=X, [1]=Y, [2]=S
    joint = d.pmf.reshape(n_x, n_y, n_s)
    ps = joint.sum(axis=(0, 1))
    px_given_s = np.zeros((n_x, n_s))
    py_given_s = np.zeros((n_y, n_s))
    for s in range(n_s):
        slice_s = joint[:, :, s]
        total = slice_s.sum()
        if total > 0:
            px_given_s[:, s] = slice_s.sum(axis=1) / total
            py_given_s[:, s] = slice_s.sum(axis=0) / total
    return px_given_s, py_given_s, ps


def admui_dist(dist, sources, target, *, eps=1e-7, maxiter=_DEFAULT_MAXITER, ip_method="IS"):
    """
    Compute the BROJA-optimal joint via admUI.

    Parameters
    ----------
    dist : Distribution
        Original distribution.
    sources : list
        Two source variable groups.
    target : list
        Target variable group.

    Returns
    -------
    q_dist : Distribution
        Optimal Q on coalesced (source0, source1, target).
    meta : dict
        ``converged``, ``n_iters``.
    """
    d = broja_prepare_dist(dist, sources, target)
    px_given_s, py_given_s, ps = _conditionals_from_dist(d)
    q_sxy, n_iters, converged = admui_optimize(
        px_given_s, py_given_s, ps, eps=eps, maxiter=maxiter, ip_method=ip_method
    )
    # q_sxy is (n_s, n_x, n_y); dit layout is (n_x, n_y, n_s)
    q_xyz = q_sxy.transpose(1, 2, 0).reshape(-1)
    q_dist = d.copy()
    q_dist.pmf = q_xyz
    meta = {"converged": converged, "n_iters": n_iters}
    return q_dist, meta
