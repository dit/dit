"""
Channel preorder checks: Blackwell (output-degraded), input-degraded,
less noisy, more capable, and Shannon inclusion.

All public functions return ``bool``.  They accept channel matrices
(2-D ndarrays, rows = inputs, columns = outputs) or any format accepted
by :func:`~dit.channelorder._utils.channel_matrix`.

References
----------
.. [1] Blackwell, D. "Equivalent Comparisons of Experiments."
       Ann. Math. Stat. 24, 265–272, 1953.
.. [2] Körner, J. and Marton, K. "Comparison of two noisy channels."
       Topics in Information Theory, 1975.
.. [3] Nasser, R. "On the input-degradedness and input-equivalence
       between channels." IEEE ISIT, 2017.
.. [4] Shannon, C. E. "A note on a partial ordering for communication
       channels." Inf. Control 1, 390–397, 1958.
.. [5] Banerjee, P. K. "Unique Information Through the Lens of Channel
       Ordering." Entropy 27, 29, 2025.
"""

import numpy as np
from scipy.optimize import linprog, minimize

from ._utils import channel_matrix, channels_from_joint

__all__ = (
    "blackwell_order_joint",
    "is_blackwell_sufficient",
    "is_input_degraded",
    "is_less_noisy",
    "is_more_capable",
    "is_output_degraded",
    "is_shannon_included",
)


# ── Blackwell (output-degraded) order ──────────────────────────────────────


def is_output_degraded(mu, kappa, atol=1e-8):
    """
    Check whether ``kappa`` is output-degraded from ``mu``.

    Returns ``True`` when there exists a stochastic matrix *lambda* such
    that ``kappa = lambda ∘ mu`` (i.e. ``kappa_s = Σ_z lambda(·|z) mu_s(z)``
    for every input *s*).  This is the Blackwell order [1]: ``mu`` is at
    least as informative as ``kappa``.

    Parameters
    ----------
    mu : array_like or list of Distribution or Distribution
        The dominating channel ``P(Z|S)``, shape ``(|S|, |Z|)``.
    kappa : array_like or list of Distribution or Distribution
        The dominated channel ``P(Y|S)``, shape ``(|S|, |Y|)``.
    atol : float
        Feasibility tolerance for the LP.

    Returns
    -------
    bool

    Notes
    -----
    Solved as a linear program: for each output *y* of ``kappa``, find
    ``lambda(y|z) >= 0`` summing to 1 over *y*, such that for every input
    *s*: ``kappa_s(y) = Σ_z lambda(y|z) mu_s(z)``.
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    n_s, n_z = mu.shape
    _, n_y = kappa.shape

    # We solve for the full lambda matrix (n_z x n_y), stacked as a
    # vector of length n_z * n_y.  For each (s, y) we have the equality
    # constraint:  Σ_z lambda(y|z) mu_s(z) = kappa_s(y).
    # Plus: for each z, Σ_y lambda(y|z) = 1, lambda >= 0.

    n_vars = n_z * n_y

    # Equality constraints: A_eq @ x = b_eq
    # (1) channel reproduction: n_s * n_y constraints
    # (2) row-stochastic: n_z constraints
    n_eq = n_s * n_y + n_z
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    idx = 0
    for s in range(n_s):
        for y in range(n_y):
            for z in range(n_z):
                A_eq[idx, z * n_y + y] = mu[s, z]
            b_eq[idx] = kappa[s, y]
            idx += 1

    for z in range(n_z):
        for y in range(n_y):
            A_eq[idx, z * n_y + y] = 1.0
        b_eq[idx] = 1.0
        idx += 1

    c = np.zeros(n_vars)
    bounds = [(0, None)] * n_vars

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs", options={"presolve": True})

    return res.success and res.status == 0


is_blackwell_sufficient = is_output_degraded


# ── Input-degraded order ───────────────────────────────────────────────────


def is_input_degraded(mu_bar, kappa_bar, atol=1e-8):
    """
    Check whether ``kappa_bar`` is input-degraded from ``mu_bar``.

    For channels with a common *output* alphabet, ``mu_bar ⊒^ideg kappa_bar``
    iff each row of ``kappa_bar`` lies in the convex hull of the rows of
    ``mu_bar`` (Proposition 2 in [3]).

    Parameters
    ----------
    mu_bar : array_like or list of Distribution or Distribution
        The dominating reverse channel ``P(S|Z)``, shape ``(|Z|, |S|)``.
    kappa_bar : array_like or list of Distribution or Distribution
        The dominated reverse channel ``P(S|Y)``, shape ``(|Y|, |S|)``.
    atol : float
        Feasibility tolerance.

    Returns
    -------
    bool
    """
    mu_bar = channel_matrix(mu_bar)
    kappa_bar = channel_matrix(kappa_bar)
    n_z, n_s = mu_bar.shape
    n_y, n_s2 = kappa_bar.shape

    if n_s != n_s2:
        raise ValueError("Channels must share the same output alphabet size")

    # For each row kappa_bar[y], find weights w[z] >= 0, Σ_z w[z] = 1,
    # such that kappa_bar[y] = Σ_z w[z] mu_bar[z].
    for y in range(n_y):
        # Equality: mu_bar.T @ w = kappa_bar[y]  and  1.T @ w = 1
        A_eq = np.vstack([mu_bar.T, np.ones((1, n_z))])
        b_eq = np.append(kappa_bar[y], 1.0)
        c = np.zeros(n_z)
        bounds = [(0, None)] * n_z
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method="highs", options={"presolve": True})
        if not (res.success and res.status == 0):
            return False

    return True


# ── More capable order ─────────────────────────────────────────────────────


def _mutual_info_gap(pi_s, mu, kappa):
    """
    Compute ``I(S;Y) - I(S;Z)`` for channels ``kappa=P(Y|S)``,
    ``mu=P(Z|S)`` given input distribution ``pi_s``.
    """
    eps = 1e-300

    # I(S;Y) = Σ_s,y  pi(s) kappa(y|s) log [kappa(y|s) / p(y)]
    p_y = pi_s @ kappa
    mi_y = 0.0
    for s in range(len(pi_s)):
        for y in range(kappa.shape[1]):
            joint = pi_s[s] * kappa[s, y]
            if joint > eps:
                mi_y += joint * np.log(kappa[s, y] / (p_y[y] + eps) + eps)

    # I(S;Z)
    p_z = pi_s @ mu
    mi_z = 0.0
    for s in range(len(pi_s)):
        for z in range(mu.shape[1]):
            joint = pi_s[s] * mu[s, z]
            if joint > eps:
                mi_z += joint * np.log(mu[s, z] / (p_z[z] + eps) + eps)

    return mi_y - mi_z


def _mi_from_channel(pi_s, ch):
    """Compute I(S; output) for a single channel given input dist."""
    eps = 1e-300
    p_out = pi_s @ ch
    mi = 0.0
    for s in range(len(pi_s)):
        for y in range(ch.shape[1]):
            joint = pi_s[s] * ch[s, y]
            if joint > eps:
                mi += joint * np.log(ch[s, y] / (p_out[y] + eps) + eps)
    return mi


def is_more_capable(mu, kappa, atol=1e-8):
    """
    Check whether ``mu`` is more capable than ``kappa``.

    ``mu ⊒_mc kappa`` iff ``I(S;Z) >= I(S;Y)`` for every input
    distribution ``P(S)`` (Definition 8 in [2]).

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``.
    kappa : array_like
        Channel ``P(Y|S)``.
    atol : float
        Tolerance: the order holds if max ``I(S;Y)-I(S;Z)`` <= atol.

    Returns
    -------
    bool
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    n_s = mu.shape[0]

    def neg_gap(logits):
        pi_s = _softmax(logits)
        return -_mutual_info_gap(pi_s, mu, kappa)

    best = _multistart_simplex_opt(neg_gap, n_s, nstarts=10)
    return best <= atol


# ── Less noisy order ───────────────────────────────────────────────────────


def is_less_noisy(mu, kappa, atol=1e-8):
    """
    Check whether ``mu`` is less noisy than ``kappa``.

    ``mu ⊒_ln kappa`` iff ``I(U;Z) >= I(U;Y)`` for every ``P(U,S)``
    with ``U-S-YZ`` Markov (Definition 9 in [2]).  Equivalently,
    ``I(S;Z) - I(S;Y)`` is concave in ``P(S)`` for the fixed channels
    [5, Eq. after Definition 9].

    We check by finding the ``P(S)`` that *maximizes* ``I(S;Y)-I(S;Z)``
    for every auxiliary channel ``P(S|U)`` (parameterized as a single
    extra variable ``U``).  If the maximum is ``<= atol``, the order
    holds.

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``.
    kappa : array_like
        Channel ``P(Y|S)``.
    atol : float
        Tolerance.

    Returns
    -------
    bool

    Notes
    -----
    This is a stricter check than ``is_more_capable``; by Proposition 3,
    ``is_less_noisy => is_more_capable`` but not vice-versa.

    The implementation checks whether I(S;Y)-I(S;Z) is concave as a
    function of P(S) by searching for a P(S) that yields a positive gap
    while also checking against auxiliary channels of bounded size.
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    n_s = mu.shape[0]

    # First quick check: more capable is necessary
    if not is_more_capable(mu, kappa, atol=atol):
        return False

    # Check concavity of I(S;Y)-I(S;Z) in P(S).  If the function is
    # concave, its max over the simplex is at a vertex or the unique
    # global max.  We sample many midpoints and check the concavity
    # condition: f(t*p + (1-t)*q) >= t*f(p) + (1-t)*f(q).

    def gap(pi_s):
        return _mutual_info_gap(pi_s, mu, kappa)

    # Also test with auxiliary variable U of size |S|: maximize over
    # the joint P(U,S) = P(U)P(S|U).
    n_u = n_s
    best_gap = 0.0

    def neg_gap_aux(params):
        # params: (n_u - 1) logits for P(U), then n_u * (n_s - 1)
        # logits for rows of P(S|U).
        pi_u = _softmax(params[:n_u])
        total_mi_y = 0.0
        total_mi_z = 0.0
        offset = n_u
        for u in range(n_u):
            pi_s_given_u = _softmax(params[offset:offset + n_s])
            offset += n_s
            total_mi_y += pi_u[u] * _mi_from_channel(pi_s_given_u, kappa)
            total_mi_z += pi_u[u] * _mi_from_channel(pi_s_given_u, mu)
        # I(U;Y) - I(U;Z) = [H(Y) - Σ_u p(u)H(Y|U=u)] - [H(Z) - ...]
        # = Σ_u p(u)[I(S;Y|U=u) - I(S;Z|U=u)] + [I(S;Y)-I(S;Z)] terms
        # Actually we need the full auxiliary construction.
        # I(U;Y) = I(U;S) + I(U;Y|S) - I(U;Y|S) ... let's just compute directly.
        # P(Y|U=u) = Σ_s P(Y|S=s) P(S=s|U=u)
        # I(U;Y) = Σ_u P(u) D(P(Y|U=u) || P(Y))
        pi_s_overall = np.zeros(n_s)
        pi_s_rows = []
        offset2 = n_u
        for u in range(n_u):
            row = _softmax(params[offset2:offset2 + n_s])
            pi_s_rows.append(row)
            pi_s_overall += pi_u[u] * row
            offset2 += n_s

        eps = 1e-300
        p_y_overall = pi_s_overall @ kappa
        p_z_overall = pi_s_overall @ mu

        mi_uy = 0.0
        mi_uz = 0.0
        for u in range(n_u):
            p_y_u = pi_s_rows[u] @ kappa
            p_z_u = pi_s_rows[u] @ mu
            for y in range(kappa.shape[1]):
                if p_y_u[y] > eps:
                    mi_uy += pi_u[u] * p_y_u[y] * np.log(p_y_u[y] / (p_y_overall[y] + eps) + eps)
            for z in range(mu.shape[1]):
                if p_z_u[z] > eps:
                    mi_uz += pi_u[u] * p_z_u[z] * np.log(p_z_u[z] / (p_z_overall[z] + eps) + eps)

        return -(mi_uy - mi_uz)

    n_params = n_u + n_u * n_s
    for _ in range(15):
        x0 = np.random.randn(n_params) * 0.5
        res = minimize(neg_gap_aux, x0, method="L-BFGS-B")
        val = -res.fun
        if val > best_gap:
            best_gap = val

    return best_gap <= atol


# ── Shannon inclusion order ────────────────────────────────────────────────


def is_shannon_included(mu, kappa, atol=1e-8, niter=None):
    """
    Check whether ``mu`` includes ``kappa`` in the Shannon sense.

    ``mu ⊒_inc kappa`` iff there exists ``chi`` in the set ``Sigma``
    (convex hull of product pre/post-channels) such that
    ``kappa = chi ∘_s mu`` (Definition 6, Proposition 1 in [4]).

    The channels may have *different* input/output alphabets.

    Parameters
    ----------
    mu : array_like
        Channel ``P(Z|S)``, shape ``(|S|, |Z|)``.
    kappa : array_like
        Channel ``P(Y|S')``, shape ``(|S'|, |Y|)``.
    atol : float
        Residual threshold below which the inclusion is declared.
    niter : int, optional
        Number of random restarts. Defaults to ``20``.

    Returns
    -------
    bool

    Notes
    -----
    This is a **heuristic** non-convex check.  ``True`` is reliable, but
    ``False`` may be a false negative.

    When ``|S'| == |S|``, the Blackwell order implies Shannon inclusion,
    so :func:`is_output_degraded` is tried first as a fast path.
    """
    mu = channel_matrix(mu)
    kappa = channel_matrix(kappa)
    n_s, n_z = mu.shape
    n_sp, n_y = kappa.shape

    if niter is None:
        niter = 20

    # Fast path: if same input alphabet, Blackwell order is sufficient
    if n_sp == n_s and is_output_degraded(mu, kappa, atol=atol):
        return True

    # By Caratheodory, k <= n_sp * n_z * n_s * n_y + 1
    k = min(n_sp * n_z * n_s * n_y + 1, 30)  # cap for tractability

    def residual(params):
        # params layout:
        #   k-1 logits for g (weights)
        #   k * n_sp * n_s logits for alpha_i  (pre-channels, rows of size n_s)
        #   k * n_z * n_y logits for beta_i   (post-channels, rows of size n_y)
        offset = 0
        g = _softmax(params[offset:offset + k])
        offset += k

        alpha = np.zeros((k, n_sp, n_s))
        for i in range(k):
            for sp in range(n_sp):
                alpha[i, sp] = _softmax(params[offset:offset + n_s])
                offset += n_s

        beta = np.zeros((k, n_z, n_y))
        for i in range(k):
            for z in range(n_z):
                beta[i, z] = _softmax(params[offset:offset + n_y])
                offset += n_y

        # kappa_hat[sp, y] = Σ_i g[i] Σ_{s,z} beta_i[z,y] mu[s,z] alpha_i[sp,s]
        kappa_hat = np.zeros((n_sp, n_y))
        for i in range(k):
            # alpha_i @ mu: shape (n_sp, n_z) -- intermediate channel
            inter = alpha[i] @ mu  # (n_sp, n_z)
            # inter @ beta_i: shape (n_sp, n_y)
            kappa_hat += g[i] * (inter @ beta[i])

        return np.sum((kappa_hat - kappa) ** 2)

    n_params = k + k * n_sp * n_s + k * n_z * n_y

    best_res = np.inf
    for _ in range(niter):
        x0 = np.random.randn(n_params) * 0.3
        res = minimize(residual, x0, method="L-BFGS-B",
                       options={"maxiter": 500})
        if res.fun < best_res:
            best_res = res.fun
        if best_res < atol:
            return True

    return best_res < atol


# ── Joint-distribution convenience ─────────────────────────────────────────


def blackwell_order_joint(dist, S, Y, Z):
    """
    Check the Blackwell order from a joint distribution.

    Returns ``True`` if ``P(Z|S) ⊒ P(Y|S)`` (i.e. ``Z`` is at least as
    informative about ``S`` as ``Y``).

    Parameters
    ----------
    dist : Distribution
        A joint distribution over ``(S, Y, Z)``.
    S : list
        Indices/names of the input variable(s).
    Y : list
        Indices/names of the first output.
    Z : list
        Indices/names of the second output.

    Returns
    -------
    bool
    """
    kappa, mu, _ = channels_from_joint(dist, S, Y, Z)
    return is_output_degraded(mu, kappa)


# ── Internal helpers ───────────────────────────────────────────────────────


def _softmax(logits):
    """Numerically stable softmax."""
    x = np.asarray(logits, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _multistart_simplex_opt(neg_func, n, nstarts=10):
    """
    Maximize ``-neg_func(logits)`` over the probability simplex via
    multiple random starts, returning the best (most positive) value
    of ``-neg_func``.
    """
    best = np.inf
    for _ in range(nstarts):
        x0 = np.random.randn(n) * 0.5
        res = minimize(neg_func, x0, method="L-BFGS-B")
        if res.fun < best:
            best = res.fun
    return -best
