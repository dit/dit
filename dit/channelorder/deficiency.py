"""
Le Cam deficiencies and KL deficiencies between channels.

These quantities measure how far two channels are from being
comparable in the Blackwell (output-degraded) or input-degraded orders.

References
----------
.. [1] Le Cam, L. "Sufficiency and approximate sufficiency."
       Ann. Math. Stat. 35, 1419–1455, 1964.
.. [2] Raginsky, M. "Shannon meets Blackwell and Le Cam." IEEE ISIT, 2011.
.. [3] Banerjee, P. K. et al. "Unique informations and deficiencies."
       Allerton Conference, 2018.
.. [4] Banerjee, P. K. "Unique Information Through the Lens of Channel
       Ordering." Entropy 27, 29, 2025.
"""

import numpy as np
from scipy.optimize import linprog, minimize

from ._utils import channel_matrix, channels_from_joint

__all__ = (
    "le_cam_deficiency",
    "le_cam_distance",
    "output_kl_deficiency",
    "weighted_input_kl_deficiency",
    "weighted_le_cam_deficiency",
    "weighted_output_kl_deficiency",
    "weighted_output_kl_deficiency_joint",
)


# ── Le Cam (TV) deficiencies ───────────────────────────────────────────────


def le_cam_deficiency(mu, kappa):
    r"""
    Le Cam deficiency of ``mu`` with respect to ``kappa``.

    .. math::
        \delta(\mu, \kappa)
            = \inf_{\lambda \in \mathcal{M}(Z;Y)}
              \sup_{s \in S}
              \|\lambda \circ \mu_s - \kappa_s\|_{\mathrm{TV}}

    Equals zero iff ``mu`` output-degrades to ``kappa`` (Blackwell order).

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``, shape ``(|S|, |Z|)``.
    kappa : array_like
        Channel ``P(Y|S)``, shape ``(|S|, |Y|)``.

    Returns
    -------
    float
        Non-negative deficiency value.

    Notes
    -----
    Formulated as a linear program with variables ``lambda(y|z)``
    and ``epsilon`` (the worst-case TV distance).
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    n_s, n_z = mu.shape
    _, n_y = kappa.shape

    # Variables: [lambda(z=0,y=0), ..., lambda(z=n_z-1,y=n_y-1), epsilon,
    #             t_plus(s=0,y=0), ..., t_plus(s=n_s-1,y=n_y-1)]
    # where t_plus[s,y] >= |Σ_z lambda(y|z)mu_s(z) - kappa_s(y)|.
    # TV(s) = 0.5 * Σ_y (t_plus) <= epsilon.

    n_lam = n_z * n_y
    n_t = n_s * n_y
    n_vars = n_lam + 1 + n_t  # lambda, epsilon, t_plus

    idx_lam = lambda z, y: z * n_y + y  # noqa: E731
    idx_eps = n_lam
    idx_t = lambda s, y: n_lam + 1 + s * n_y + y  # noqa: E731

    c = np.zeros(n_vars)
    c[idx_eps] = 1.0  # minimize epsilon

    # Inequality constraints: A_ub @ x <= b_ub
    rows_ub = []
    b_ub_list = []

    # For each (s, y):
    #   Σ_z lambda(y|z) mu_s(z) - kappa_s(y) <= t[s,y]
    #   -(Σ_z lambda(y|z) mu_s(z) - kappa_s(y)) <= t[s,y]
    for s in range(n_s):
        for y in range(n_y):
            # diff <= t
            row = np.zeros(n_vars)
            for z in range(n_z):
                row[idx_lam(z, y)] = mu[s, z]
            row[idx_t(s, y)] = -1.0
            rows_ub.append(row)
            b_ub_list.append(kappa[s, y])

            # -diff <= t
            row2 = np.zeros(n_vars)
            for z in range(n_z):
                row2[idx_lam(z, y)] = -mu[s, z]
            row2[idx_t(s, y)] = -1.0
            rows_ub.append(row2)
            b_ub_list.append(-kappa[s, y])

    # For each s: 0.5 * Σ_y t[s,y] <= epsilon
    for s in range(n_s):
        row = np.zeros(n_vars)
        for y in range(n_y):
            row[idx_t(s, y)] = 0.5
        row[idx_eps] = -1.0
        rows_ub.append(row)
        b_ub_list.append(0.0)

    A_ub = np.array(rows_ub)
    b_ub = np.array(b_ub_list)

    # Equality: for each z, Σ_y lambda(y|z) = 1
    A_eq = np.zeros((n_z, n_vars))
    b_eq = np.ones(n_z)
    for z in range(n_z):
        for y in range(n_y):
            A_eq[z, idx_lam(z, y)] = 1.0

    bounds = [(0, None)] * n_lam + [(0, None)] + [(0, None)] * n_t

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        return max(0.0, res.fun)
    return np.inf  # pragma: no cover


def weighted_le_cam_deficiency(mu, kappa, pi):
    r"""
    Weighted Le Cam deficiency of ``mu`` w.r.t. ``kappa``.

    .. math::
        \delta_\pi(\mu, \kappa)
            = \inf_{\lambda} \mathbb{E}_{s \sim \pi}
              \|\lambda \circ \mu_s - \kappa_s\|_{\mathrm{TV}}

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``, shape ``(|S|, |Z|)``.
    kappa : array_like
        Channel ``P(Y|S)``, shape ``(|S|, |Y|)``.
    pi : array_like
        Input marginal ``P(S)``, length ``|S|``.

    Returns
    -------
    float
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    pi = np.asarray(pi, dtype=float)
    n_s, n_z = mu.shape
    _, n_y = kappa.shape

    n_lam = n_z * n_y
    n_t = n_s * n_y
    n_vars = n_lam + n_t

    idx_lam = lambda z, y: z * n_y + y  # noqa: E731
    idx_t = lambda s, y: n_lam + s * n_y + y  # noqa: E731

    # Minimize Σ_s pi(s) * 0.5 * Σ_y t[s,y]
    c = np.zeros(n_vars)
    for s in range(n_s):
        for y in range(n_y):
            c[idx_t(s, y)] = 0.5 * pi[s]

    rows_ub = []
    b_ub_list = []

    for s in range(n_s):
        for y in range(n_y):
            row = np.zeros(n_vars)
            for z in range(n_z):
                row[idx_lam(z, y)] = mu[s, z]
            row[idx_t(s, y)] = -1.0
            rows_ub.append(row)
            b_ub_list.append(kappa[s, y])

            row2 = np.zeros(n_vars)
            for z in range(n_z):
                row2[idx_lam(z, y)] = -mu[s, z]
            row2[idx_t(s, y)] = -1.0
            rows_ub.append(row2)
            b_ub_list.append(-kappa[s, y])

    A_ub = np.array(rows_ub) if rows_ub else None
    b_ub = np.array(b_ub_list) if b_ub_list else None

    A_eq = np.zeros((n_z, n_vars))
    b_eq = np.ones(n_z)
    for z in range(n_z):
        for y in range(n_y):
            A_eq[z, idx_lam(z, y)] = 1.0

    bounds = [(0, None)] * n_lam + [(0, None)] * n_t

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        return max(0.0, res.fun)
    return np.inf  # pragma: no cover


# ── Le Cam distance (pseudometric) ────────────────────────────────────────


def le_cam_distance(mu, kappa):
    r"""
    Le Cam distance between ``mu`` and ``kappa``.

    .. math::
        \Delta(\mu, \kappa)
            = \max\bigl(\delta(\mu, \kappa),\; \delta(\kappa, \mu)\bigr)

    This is a pseudometric on channels with a common input alphabet.

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``.
    kappa : array_like
        Channel ``P(Y|S)``.

    Returns
    -------
    float
    """
    return max(le_cam_deficiency(mu, kappa), le_cam_deficiency(kappa, mu))


# ── KL deficiencies ───────────────────────────────────────────────────────


def _kl_divergence_vec(p, q):
    """D(p || q) for probability vectors, returning +inf when needed."""
    eps = 1e-300
    result = 0.0
    for i in range(len(p)):
        if p[i] > eps:
            if q[i] < eps:
                return np.inf
            result += p[i] * np.log(p[i] / q[i])
    return result


def output_kl_deficiency(mu, kappa):
    r"""
    Output KL deficiency of ``mu`` with respect to ``kappa``.

    .. math::
        \delta_o(\mu, \kappa)
            = \inf_{\lambda \in \mathcal{M}(Z;Y)}
              \sup_{s \in S}
              D(\kappa_s \| \lambda \circ \mu_s)

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``, shape ``(|S|, |Z|)``.
    kappa : array_like
        Channel ``P(Y|S)``, shape ``(|S|, |Y|)``.

    Returns
    -------
    float

    Notes
    -----
    Solved as a minimax problem via multi-start convex optimization
    over the stochastic matrix ``lambda``.
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    n_s, n_z = mu.shape
    _, n_y = kappa.shape

    def objective(params):
        lam = _params_to_stochastic(params, n_z, n_y)
        worst = 0.0
        for s in range(n_s):
            q = mu[s] @ lam  # (n_y,)
            kl = _kl_divergence_vec(kappa[s], q)
            if kl > worst:
                worst = kl
        return worst

    return _optimize_over_stochastic(objective, n_z, n_y, nstarts=15)


def weighted_output_kl_deficiency(mu, kappa, pi):
    r"""
    Weighted output KL deficiency of ``mu`` w.r.t. ``kappa``.

    .. math::
        \delta^\pi_o(\mu, \kappa)
            = \min_{\lambda \in \mathcal{M}(Z;Y)}
              D(\kappa \| \lambda \circ \mu \mid \pi_S)
            = \min_\lambda \sum_s \pi(s)\,
              D(\kappa_s \| \lambda \circ \mu_s)

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``, shape ``(|S|, |Z|)``.
    kappa : array_like
        Channel ``P(Y|S)``, shape ``(|S|, |Y|)``.
    pi : array_like
        Input marginal ``P(S)``, length ``|S|``.

    Returns
    -------
    float

    Notes
    -----
    Convex in ``lambda`` and solved via multi-start optimization.
    Related to the Blackwell order: equals zero iff ``mu`` output-degrades
    to ``kappa``.
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    pi = np.asarray(pi, dtype=float)
    n_s, n_z = mu.shape
    _, n_y = kappa.shape

    def objective(params):
        lam = _params_to_stochastic(params, n_z, n_y)
        total = 0.0
        for s in range(n_s):
            if pi[s] < 1e-300:
                continue
            q = mu[s] @ lam
            kl = _kl_divergence_vec(kappa[s], q)
            total += pi[s] * kl
        return total

    return _optimize_over_stochastic(objective, n_z, n_y, nstarts=15)


def weighted_input_kl_deficiency(mu_bar, kappa_bar, pi):
    r"""
    Weighted input KL deficiency of ``mu_bar`` w.r.t. ``kappa_bar``.

    .. math::
        \delta^\pi_i(\bar\mu, \bar\kappa)
            = \min_{\bar\lambda \in \mathcal{M}(Y;Z)}
              \sum_y \pi(y)\,
              D\bigl(\bar\kappa_y \| \sum_z \bar\lambda(z|y)\,\bar\mu_z\bigr)

    For channels with a common *output* alphabet, this quantifies the
    cost of approximating ``kappa_bar`` from ``mu_bar`` via input
    randomization.

    Parameters
    ----------
    mu_bar : array_like
        Reverse channel ``P(S|Z)``, shape ``(|Z|, |S|)``.
    kappa_bar : array_like
        Reverse channel ``P(S|Y)``, shape ``(|Y|, |S|)``.
    pi : array_like
        Marginal ``P(Y)``, length ``|Y|``.

    Returns
    -------
    float
    """
    mu_bar = channel_matrix(mu_bar)
    kappa_bar = channel_matrix(kappa_bar)
    pi = np.asarray(pi, dtype=float)
    n_z, n_s = mu_bar.shape
    n_y, _ = kappa_bar.shape

    def objective(params):
        # lambda_bar: P(Z|Y), shape (n_y, n_z)
        lam_bar = _params_to_stochastic(params, n_y, n_z)
        total = 0.0
        for y in range(n_y):
            if pi[y] < 1e-300:
                continue
            q = lam_bar[y] @ mu_bar  # (n_s,)
            kl = _kl_divergence_vec(kappa_bar[y], q)
            total += pi[y] * kl
        return total

    return _optimize_over_stochastic(objective, n_y, n_z, nstarts=15)


# ── Joint distribution convenience wrappers ────────────────────────────────


def weighted_output_kl_deficiency_joint(dist, S, Y, Z):
    """
    Weighted output KL deficiency from a joint distribution.

    Extracts ``P(Y|S)`` (kappa), ``P(Z|S)`` (mu), and ``P(S)`` (pi)
    from the joint, then computes the weighted output KL deficiency.

    Parameters
    ----------
    dist : Distribution
        A joint distribution.
    S : list
        Input variable indices/names.
    Y : list
        First output variable indices/names.
    Z : list
        Second output variable indices/names.

    Returns
    -------
    float
        ``delta^pi_o(mu, kappa)`` -- cost of approximating ``P(Y|S)``
        from ``P(Z|S)`` via output randomization.
    """
    kappa, mu, pi_s = channels_from_joint(dist, S, Y, Z)
    return weighted_output_kl_deficiency(mu, kappa, pi_s)


# ── Internal helpers ───────────────────────────────────────────────────────


def _params_to_stochastic(params, n_rows, n_cols):
    """
    Map an unconstrained parameter vector to a row-stochastic matrix
    via softmax on each row.
    """
    mat = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        logits = params[r * n_cols : (r + 1) * n_cols]
        x = logits - logits.max()
        e = np.exp(x)
        mat[r] = e / e.sum()
    return mat


def _optimize_over_stochastic(objective, n_rows, n_cols, nstarts=15):
    """
    Minimize ``objective(params)`` where ``params`` is mapped to a
    row-stochastic matrix via :func:`_params_to_stochastic`.
    """
    n_params = n_rows * n_cols
    best = np.inf
    for _ in range(nstarts):
        x0 = np.random.randn(n_params) * 0.3
        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 500})
        if res.fun < best:
            best = res.fun
    return max(0.0, best)
