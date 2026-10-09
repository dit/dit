"""
The less-noisy intersection information I_ln^∩ from Gomes & Figueiredo (2023).

Defines redundancy as:

    I_ln^∩(Y1, ..., Yn → T) = max_{Q : Q ≤_ln Y_i ∀i} I(Q; T)

where ≤_ln is the "less noisy" channel preorder: K_Q ≤_ln K^(i) iff
I(U; Q) ≤ I(U; Y_i) for every U with U - T - (Q, Y_i). Equivalently,
the one-way secret key rate from T to Q with Y_i eavesdropping vanishes.

By van Dijk (1997), K_Q ≤_ln K^(i) iff I(T; Y_i) - I(T; Q) is concave in
P(T). Since the Hessian of H(p K) in p is -sum_j k_j k_j^T / (p . k_j), with
k_j the columns of K, this holds iff for every p in the simplex and every
tangent direction v,

    sum_j (v . k^Q_j)^2 / (p . k^Q_j)  <=  sum_j (v . k^(i)_j)^2 / (p . k^(i)_j).

We impose these constraints at sampled (p, v), maximize I(Q; T) with SLSQP,
and keep the best candidate passing an exact-in-v check on a denser sample of
p. No bound on |Q| is known; it defaults to max_i |Y_i|.

References
----------
.. [1] A. F. C. Gomes and M. A. T. Figueiredo, "Orders between Channels
       and Implications for Partial Information Decomposition",
       Entropy 25, 975, 2023.
.. [2] M. van Dijk, "On a special class of broadcast channels with
       confidential messages", IEEE Trans. Inf. Theory 43(2), 712-714, 1997.
"""

import numpy as np
from scipy.optimize import minimize

from ...channelorder._utils import channels_from_joint
from ..pid import BaseBivariatePID
from .imc import _mi_bits, _mi_grad_wrt_K, _params_to_stochastic, _softmax_vjp

__all__ = ("PID_LN",)


def _interior_points(n_t, n_points, rng):
    """Interior simplex points, dense near the boundary where curvature concentrates."""
    if n_t == 2:
        u = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_points + 2)[1:-1])
        return np.c_[u, 1 - u]
    raw = rng.exponential(size=(n_points, n_t)) ** rng.choice([1.0, 3.0], size=(n_points, 1))
    pts = raw / raw.sum(axis=1, keepdims=True)
    return 0.999 * pts + 0.001 / n_t


def _tangent_basis(n_t):
    """Orthonormal basis of {v : sum(v) = 0}."""
    basis, _ = np.linalg.qr(np.eye(n_t) - 1.0 / n_t)
    return basis[:, : n_t - 1]


def _directions(n_t, n_random, rng):
    """Tangent directions: all e_a - e_b plus random ones."""
    dirs = [np.eye(n_t)[a] - np.eye(n_t)[b] for a in range(n_t) for b in range(a + 1, n_t)]
    if n_t > 2 and n_random > 0:
        rand = rng.standard_normal((n_random, n_t))
        dirs.extend(rand - rand.mean(axis=1, keepdims=True))
    dirs = np.array(dirs)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _curvature(K, P, V):
    """sum_j (v . k_j)^2 / (p . k_j) for each row pair (p, v)."""
    s = V @ K
    r = P @ K
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, s**2 / r, 0.0).sum(axis=1)


def _curvature_grad(K, P, V):
    """Gradient of :func:`_curvature` w.r.t. K, shape (rows, |T|, |Q|)."""
    s = V @ K
    r = P @ K
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(r > 0, 2 * s / r, 0.0)
        b = np.where(r > 0, s**2 / r**2, 0.0)
    return a[:, None, :] * V[:, :, None] - b[:, None, :] * P[:, :, None]


def _is_dominated(k_q, channels, points, atol=1e-6):
    """Exact-in-v check that every channel is less noisy than k_q at each sampled point."""
    T = _tangent_basis(k_q.shape[0])

    def forms(K, p):
        r = p @ K
        keep = r > 0
        return T.T @ (K[:, keep] / r[keep]) @ K[:, keep].T @ T

    for p in points:
        dq = forms(k_q, p)
        for ch in channels:
            di = forms(ch, p)
            if np.linalg.eigvalsh(di - dq).min() < -atol * (1 + np.abs(di).max()):
                return False
    return True


def _tied_rows(channels, atol=1e-12):
    """
    One-hot (|T|, classes) matrix grouping target values whose rows coincide in
    some source channel. On that edge of the simplex the source has zero
    curvature, so every feasible K_Q must have equal rows there as well.
    """
    n_t = channels[0].shape[0]
    parent = list(range(n_t))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for ch in channels:
        for a in range(n_t):
            for b in range(a + 1, n_t):
                if np.abs(ch[a] - ch[b]).max() <= atol:
                    parent[find(b)] = find(a)
    roots = sorted({find(a) for a in range(n_t)})
    E = np.zeros((n_t, len(roots)))
    for a in range(n_t):
        E[a, roots.index(find(a))] = 1
    return E


def _make_feasible(k_q, channels, pi_t, points, steps=30):
    """
    Mix ``k_q`` with the constant channel at its output marginal, as little as
    the dense check allows. Mixing is a garbling, which can only shrink the
    curvature, so the fully mixed channel is always feasible.
    """
    if _is_dominated(k_q, channels, points):
        return k_q
    flat = np.tile(pi_t @ k_q, (k_q.shape[0], 1))
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if _is_dominated((1 - mid) * k_q + mid * flat, channels, points):
            hi = mid
        else:
            lo = mid
    return (1 - hi) * k_q + hi * flat


def _less_noisy_ii(channels, pi_t, n_q=None, n_points=None, niter=None, seed=None):
    """
    Compute I_ln^∩ via sampled less-noisy curvature constraints.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).
    n_q : int or None
        Cardinality of Q.  Defaults to max(|Y_i|).
    n_points : int or None
        Number of sampled points of the simplex for the constraints.
    niter : int or None
        Number of optimization starts.
    seed : int or None
        Random seed for the sampled points and starts.

    Returns
    -------
    float
        The largest I(Q; T) found among candidates passing the dense check.
    """
    n_t = channels[0].shape[0]
    if n_q is None:
        n_q = max(ch.shape[1] for ch in channels)
    if n_points is None:
        n_points = 60 if n_t == 2 else 40 * n_t
    if niter is None:
        niter = 10

    rng = np.random.default_rng(seed)
    pts = _interior_points(n_t, n_points, rng)
    dirs = _directions(n_t, 2 * n_t, rng)
    P = np.repeat(pts, len(dirs), axis=0)
    V = np.tile(dirs, (len(pts), 1))
    bound = np.min([_curvature(ch, P, V) for ch in channels], axis=0)
    scale = np.maximum(bound, 1e-6)
    check_pts = _interior_points(n_t, 4 * n_points, rng)

    best_mi = 0.0
    for ch in channels:
        if _is_dominated(ch, channels, check_pts):
            best_mi = max(best_mi, _mi_bits(pi_t, ch))

    E = _tied_rows(channels)
    n_c = E.shape[1]

    def channel(params):
        return E @ _params_to_stochastic(params, n_c, n_q)

    def objective(params):
        return -_mi_bits(pi_t, channel(params))

    def objective_jac(params):
        rows = _params_to_stochastic(params, n_c, n_q)
        return _softmax_vjp(rows, -E.T @ _mi_grad_wrt_K(pi_t, E @ rows))

    def constraint(params):
        return (bound - _curvature(channel(params), P, V)) / scale

    def constraint_jac(params):
        rows = _params_to_stochastic(params, n_c, n_q)
        G = np.einsum("tc,ktq->kcq", E, -_curvature_grad(E @ rows, P, V) / scale[:, None, None])
        inner = (G * rows[None]).sum(axis=2, keepdims=True)
        return (rows[None] * (G - inner)).reshape(len(G), -1)

    first = E.argmax(axis=0)
    starts = []
    for ch in channels:
        padded = np.full((n_c, n_q), 1e-10)
        padded[:, : min(n_q, ch.shape[1])] = ch[first, :n_q]
        starts.append(np.log(padded + 1e-10).ravel())
    starts += [rng.standard_normal(n_c * n_q) * 0.5 for _ in range(max(1, niter - len(starts)))]

    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=objective_jac,
            constraints=[{"type": "ineq", "fun": constraint, "jac": constraint_jac}],
            options={"maxiter": 300, "ftol": 1e-12},
        )
        k_q = channel(res.x)
        if _mi_bits(pi_t, k_q) > best_mi:
            best_mi = max(best_mi, _mi_bits(pi_t, _make_feasible(k_q, channels, pi_t, check_pts)))

    return best_mi


class PID_LN(BaseBivariatePID):
    """
    Less-noisy intersection information I_ln^∩ from Gomes & Figueiredo.

    Redundancy is the maximum I(Q; T) over channels K_Q such that every
    source channel is less noisy than K_Q; equivalently, no one-way secret key
    can be agreed from T to Q against any single source.

    References
    ----------
    .. [1] A. F. C. Gomes and M. A. T. Figueiredo, "Orders between
           Channels and Implications for Partial Information
           Decomposition", Entropy 25, 975, 2023.
    """

    _name = "I_ln∩"

    @staticmethod
    def _measure(d, sources, target, n_points=None, niter=None, seed=None):
        """
        Compute the less-noisy II redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.
        n_points : int or None
            Number of sampled simplex points for the curvature constraints.
        niter : int or None
            Number of optimization starts.
        seed : int or None
            Random seed.

        Returns
        -------
        float
            The less-noisy II redundancy, in bits.
        """
        source_a, source_b = sources

        d_coal = d.coalesce([list(source_a), list(source_b), list(target)])
        src_a, src_b, tgt = d_coal.dims

        kappa_a, kappa_b, pi_t = channels_from_joint(d_coal, [tgt], [src_a], [src_b])

        return _less_noisy_ii([kappa_a, kappa_b], pi_t, n_points=n_points, niter=niter, seed=seed)
