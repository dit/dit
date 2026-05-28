"""
The Salamatian-Cohen-Médard Maximum Entropy Function.

An approximate Gács-Körner common information that allows a small helper
rate.  See Salamatian, Cohen, & Médard, "Maximum Entropy Functions:
Approximate Gács-Körner for Distributed Compression", Information Theory
and Applications Workshop (ITA), 2016 (arXiv:1604.03877).
"""

import numpy as np

from ...exceptions import ditException
from ...helpers import normalize_rvs
from ...utils import unitful

__all__ = (
    "maxent_function",
    "plot_maxent_function",
)


def _coalesced_pmf(dist, rvs):
    """Return the (n_X, n_Y) joint pmf for the two random variables."""
    d = dist.copy().coalesce(rvs)
    d.make_dense()
    return d.pmf.reshape(list(map(len, d.alphabet)))


def _q_matrix(pXY):
    """
    Build Q = D_X^{-1/2} P D_Y^{-1/2} (Witsenhausen 1975 / Salamatian §II-C).
    """
    pX = pXY.sum(axis=1)
    pY = pXY.sum(axis=0)
    Dx = np.where(pX > 0, 1.0 / np.sqrt(pX), 0.0)
    Dy = np.where(pY > 0, 1.0 / np.sqrt(pY), 0.0)
    return Dx[:, None] * pXY * Dy[None, :]


def _second_singular_vectors(pXY):
    """
    Return the second left/right singular vectors of Q.

    Subtracts the trivial rank-1 component sqrt(p_X) sqrt(p_Y)^T (which
    always carries the unit singular value of Q) before taking the SVD,
    so the leading vectors of the residual are exactly the paper's u, v
    even when Q's top singular value has multiplicity > 1 (the
    Gács-Körner-positive regime).

    Returns
    -------
    u, v : np.ndarray or None
        The (left, right) singular vectors, or (None, None) when one of
        the alphabets is degenerate (size < 2).
    """
    if min(pXY.shape) < 2:
        return None, None

    Q = _q_matrix(pXY)
    sx = np.sqrt(pXY.sum(axis=1))
    sy = np.sqrt(pXY.sum(axis=0))
    Q_resid = Q - np.outer(sx, sy)

    U, _, Vt = np.linalg.svd(Q_resid, full_matrices=False)
    return U[:, 0], Vt[0, :]


def _h2(*probs):
    """Shannon entropy (base 2) of a list of probabilities."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _eval_phi(pXY, phi_x, phi_y):
    """
    Return (H(phi_X(X)), H(phi_X(X) | phi_Y(Y))) for ±1 label vectors.
    """
    px_pos = phi_x > 0
    py_pos = phi_y > 0

    p_pp = float(pXY[np.ix_(px_pos, py_pos)].sum())
    p_pn = float(pXY[np.ix_(px_pos, ~py_pos)].sum())
    p_np = float(pXY[np.ix_(~px_pos, py_pos)].sum())
    p_nn = float(pXY[np.ix_(~px_pos, ~py_pos)].sum())

    p_x_pos = p_pp + p_pn
    p_x_neg = p_np + p_nn
    p_y_pos = p_pp + p_np
    p_y_neg = p_pn + p_nn

    H_phi_x = _h2(p_x_pos, p_x_neg)
    H_phi_y = _h2(p_y_pos, p_y_neg)
    H_joint = _h2(p_pp, p_pn, p_np, p_nn)

    H_cond = max(0.0, H_joint - H_phi_y)
    return H_phi_x, H_cond


def _threshold_sweep(pXY):
    """
    Evaluate the spectral algorithm at every distinct threshold position.

    Distinct thresholds are taken as the midpoints between sorted entries
    of the left singular vector u, padded with sentinels below the min
    and above the max so the all-plus and all-minus partitions are
    included exactly once each.

    Returns
    -------
    points : list of (t, H_phi_x, H_cond, phi_x, phi_y)
        One entry per distinct threshold, sorted by t.  Empty list if
        the alphabet is degenerate.
    """
    u, v = _second_singular_vectors(pXY)
    if u is None:
        return []

    su = np.sort(u)
    midpoints = (su[:-1] + su[1:]) / 2.0
    ts = np.concatenate(([su[0] - 1.0], midpoints, [su[-1] + 1.0]))

    points = []
    for t in ts:
        phi_x = np.where(u > t, 1, -1)
        phi_y = np.where(v > t, 1, -1)
        H_x, H_cond = _eval_phi(pXY, phi_x, phi_y)
        points.append((float(t), H_x, H_cond, phi_x.copy(), phi_y.copy()))

    return points


def _spectral_value(pXY, epsilon):
    """Max H(phi_X) over the spectral threshold sweep with H_cond <= eps."""
    points = _threshold_sweep(pXY)
    feasible = [H_x for (_, H_x, H_cond, _, _) in points if H_cond <= epsilon + 1e-12]
    return max(feasible, default=0.0)


def _iter_binary_partitions(n):
    """
    Yield each binary partition of {0..n-1} once, as a length-n ±1 vector.

    The constant +1 vector is yielded first; then phi[0] is held to +1
    to break the symmetry between phi and -phi (which give identical
    H(phi(X)) and H(phi_X|phi_Y) values).
    """
    yield np.ones(n, dtype=int)
    for mask in range(1, 1 << max(n - 1, 0)):
        phi = np.ones(n, dtype=int)
        for i in range(n - 1):
            if (mask >> i) & 1:
                phi[i + 1] = -1
        yield phi


_EXACT_MAX_PAIRS = 1 << 20  # ~1e6 partition pairs


def _exact_value(pXY, epsilon):
    """Brute-force max H(phi_X) over all binary partitions of X and Y."""
    n_x, n_y = pXY.shape
    n_pairs = (1 << max(n_x - 1, 0)) * (1 << max(n_y - 1, 0))
    if n_pairs > _EXACT_MAX_PAIRS:
        msg = (
            f"exact maxent_function refuses to enumerate {n_pairs} partition "
            f"pairs for alphabets of size {n_x} x {n_y}; use method='spectral' "
            f"or coarsen the support."
        )
        raise ditException(msg)

    partitions_y = list(_iter_binary_partitions(n_y))

    best = 0.0
    for phi_x in _iter_binary_partitions(n_x):
        for phi_y in partitions_y:
            H_x, H_cond = _eval_phi(pXY, phi_x, phi_y)
            if H_cond <= epsilon + 1e-12 and H_x > best:
                best = H_x
    return best


def _prepare(dist, rvs, crvs):
    """Validate rvs/crvs and return the (n_X, n_Y) joint pmf."""
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        msg = (
            "maxent_function does not support conditioning; "
            "Salamatian et al. 2016 is strictly bivariate, unconditional."
        )
        raise ditException(msg)
    if len(rvs) != 2:
        msg = f"maxent_function requires exactly 2 random variables, got {len(rvs)}."
        raise ditException(msg)
    return _coalesced_pmf(dist, rvs)


@unitful
def maxent_function(dist, epsilon, rvs=None, crvs=None, method="spectral"):
    """
    The Salamatian-Cohen-Médard Maximum Entropy Function:

    .. math::

        M_{\\epsilon}(X; Y) = \\max_{\\phi_X, \\phi_Y}
            H(\\phi_X(X)) \\quad \\text{s.t.} \\quad
            H(\\phi_X(X) \\mid \\phi_Y(Y)) \\le \\epsilon

    where the maximization is over binary-valued functions
    :math:`\\phi_X: \\mathcal{X} \\to \\{-1, +1\\}` and
    :math:`\\phi_Y: \\mathcal{Y} \\to \\{-1, +1\\}`.  This is an
    approximate Gács-Körner common information that admits a helper
    of rate up to :math:`\\epsilon`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which to compute the measure.
    epsilon : float
        The maximum allowed helper rate
        :math:`H(\\phi_X(X) \\mid \\phi_Y(Y))`.
    rvs : list, None
        A list selecting exactly two random variables (by index or name).
        If None, all variables of `dist` are used; `dist` must therefore
        be bivariate.
    crvs : list, None
        Not supported; passing a non-empty `crvs` raises
        :class:`ditException`.
    method : {"spectral", "exact"}
        - ``"spectral"`` (default): the paper's §IV-A approximation.
          Thresholds the second left/right singular vectors of
          :math:`Q = D_X^{-1/2} P D_Y^{-1/2}` across all distinct cuts
          and returns the largest feasible :math:`H(\\phi_X(X))`.
        - ``"exact"``: brute force over every binary partition of
          :math:`\\mathcal{X}` and :math:`\\mathcal{Y}`.  Refuses to run
          when the alphabets would produce more than ~:math:`2^{20}`
          partition pairs.

    Returns
    -------
    M : float
        The Maximum Entropy Function value (in bits when units are off).

    Raises
    ------
    ditException
        If `crvs` is provided, `rvs` does not select exactly two
        variables, the method is unknown, or `method="exact"` is requested
        on an alphabet that is too large to enumerate.
    """
    pXY = _prepare(dist, rvs, crvs)

    if method == "spectral":
        return _spectral_value(pXY, float(epsilon))
    if method == "exact":
        return _exact_value(pXY, float(epsilon))

    msg = f"Unknown method {method!r}; expected 'spectral' or 'exact'."
    raise ditException(msg)


def plot_maxent_function(dist, rvs=None, crvs=None, ax=None, **plot_kwargs):
    """
    Plot the spectral Maximum Entropy Function value :math:`H(\\phi_X(X))`
    as a function of the threshold :math:`t`.

    At each distinct threshold position on the second left singular
    vector :math:`u` of :math:`Q = D_X^{-1/2} P D_Y^{-1/2}`, the binary
    function :math:`\\phi_X(i) = \\mathrm{sign}(u_i - t)` (and similarly
    :math:`\\phi_Y` from the second right singular vector :math:`v`) is
    constructed per Salamatian et al. 2016 §IV-A, and
    :math:`H(\\phi_X(X))` is plotted at each :math:`t`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which to compute the curve.
    rvs : list, None
        A list selecting exactly two random variables (by index or name).
        If None, all variables of `dist` are used; `dist` must therefore
        be bivariate.
    crvs : list, None
        Not supported; passing a non-empty `crvs` raises
        :class:`ditException`.
    ax : matplotlib.axes.Axes, None
        The axis to draw on.  A new figure and axis are created if None.
    **plot_kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plot.

    Raises
    ------
    ditException
        If `crvs` is provided, `rvs` does not select exactly two
        variables, or the joint pmf is too degenerate for a spectral
        sweep (one of the alphabets has size < 2).
    """
    import matplotlib.pyplot as plt

    pXY = _prepare(dist, rvs, crvs)
    points = _threshold_sweep(pXY)

    if not points:
        msg = "spectral threshold sweep is empty; both alphabets must have size >= 2."
        raise ditException(msg)

    ts = [t for (t, _, _, _, _) in points]
    H_xs = [H_x for (_, H_x, _, _, _) in points]

    if ax is None:
        _, ax = plt.subplots()

    plot_kwargs.setdefault("marker", "o")
    ax.plot(ts, H_xs, **plot_kwargs)
    ax.set_xlabel(r"threshold $t$")
    ax.set_ylabel(r"$H(\phi_X(X))$ [bits]")
    return ax
