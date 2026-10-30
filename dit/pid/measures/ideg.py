"""
The degradation intersection information I_d^∩ from Kolchinsky (2022).

Defines redundancy as the maximum information a channel K_Q can carry
about T while remaining a Blackwell degradation of every source channel:

    I_d^∩(Y1, ..., Yn → T) = max_{Q : Q ≤_d Y_i ∀i} I(Q; T)

An outcome q of Q corresponds to the vector u_q = (Λ_1(q|·), ..., Λ_n(q|·)),
which lies in the polyhedral cone C = {u ≥ 0 : K^(1) u_1 = K^(i) u_i ∀i}.
The vectors of all outcomes add up to the all-ones vector, and
I(Q; T) = Σ_q f(u_q) with f convex and positively homogeneous. Splitting an
outcome along the extreme rays g_r of C therefore never decreases I(Q; T), so

    I_d^∩ = max { Σ_r λ_r f(g_r) : λ ≥ 0, Σ_r λ_r g_r = 1 },

a linear program whose solution is also an optimal Q. This formulation is due
to A. F. C. Gomes.

References
----------
.. [1] A. Kolchinsky, "A Novel Approach to the Partial Information
       Decomposition", Entropy 24, 403, 2022.
.. [2] A. F. C. Gomes and M. A. T. Figueiredo, "Orders between Channels
       and Implications for Partial Information Decomposition",
       Entropy 25, 975, 2023.
"""

from fractions import Fraction
from math import gcd, lcm

import numpy as np
from scipy.optimize import linprog

from ...channelorder._utils import channels_from_joint
from ..pid import BaseBivariatePID

__all__ = ("PID_Deg",)


def _mi_bits(pi_t, ch):
    """I(T; Q) in bits for channel ch = P(Q|T) given input dist pi_t."""
    eps = 1e-300
    p_q = pi_t @ ch
    mi = 0.0
    for t in range(len(pi_t)):
        for q in range(ch.shape[1]):
            pj = pi_t[t] * ch[t, q]
            if pj > eps:
                mi += pj * np.log2(ch[t, q] / (p_q[q] + eps))
    return mi


def _mi_grad(pi_t, ch, floor=1e-12):
    """Gradient of I(T; Q) w.r.t. the channel, with logs floored so zero entries stay finite."""
    p_q = pi_t @ ch
    return pi_t[:, None] * np.log2(np.maximum(ch, floor) / np.maximum(p_q, floor)[None, :])


def _cone_equalities(channels):
    """
    Equalities which, together with u >= 0, define the cone of common degradations.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).

    Returns
    -------
    rows : ndarray
        The rows a of the equalities a @ u = 0, i.e. K^(1) u_1 = K^(i) u_i for i > 1.
    offsets : ndarray
        Where each u_i starts in u, followed by the length of u.
    """
    n_t = channels[0].shape[0]
    offsets = np.cumsum([0] + [ch.shape[1] for ch in channels])
    rows = np.zeros(((len(channels) - 1) * n_t, offsets[-1]))
    for i in range(1, len(channels)):
        block = slice((i - 1) * n_t, i * n_t)
        rows[block, offsets[0] : offsets[1]] = channels[0]
        rows[block, offsets[i] : offsets[i + 1]] = -channels[i]
    return rows, offsets


def _extreme_rays_cdd(rows, dim):
    """
    Extreme rays of {u >= 0 : rows @ u = 0}, computed by cddlib in floating point.

    Returns None if the cone is not pointed, which only happens if cddlib fails.
    """
    import cdd

    arr = np.zeros((dim + len(rows), dim + 1))
    arr[:dim, 1:] = np.eye(dim)
    arr[dim:, 1:] = rows
    mat = cdd.matrix_from_array(arr, lin_set=set(range(dim, len(arr))), rep_type=cdd.RepType.INEQUALITY)
    gens = cdd.copy_generators(cdd.polyhedron_from_matrix(mat))
    if gens.lin_set:
        return None
    return np.array([gen[1:] for gen in gens.array if gen[0] == 0], dtype=float).reshape(-1, dim)


def _extreme_rays_exact(channels):
    """
    Extreme rays of the cone of common degradations by the double description method, in
    exact integer arithmetic.

    The (exactly represented) floating-point rows of each channel are first normalized
    exactly, so that the constant degradation (the all-ones vector) lies in the cone.
    Starting from the unit vectors, which generate the orthant, each equality a @ u = 0
    keeps the rays on its hyperplane and adds, for each adjacent pair of rays on opposite
    sides of it, the ray where the face they span meets the hyperplane. Two rays are
    adjacent if no other ray vanishes on every coordinate on which both vanish.
    """
    exact = [[[Fraction(x) for x in row] for row in ch] for ch in channels]
    exact = [[[x / sum(row) for x in row] for row in ch] for ch in exact]
    sizes = [len(ch[0]) for ch in exact]
    dim = sum(sizes)
    equalities = []
    for i in range(1, len(exact)):
        start = sum(sizes[:i])
        for row_0, row_i in zip(exact[0], exact[i], strict=True):
            row = [Fraction(0)] * dim
            row[: sizes[0]] = row_0
            row[start : start + sizes[i]] = [-x for x in row_i]
            scale = lcm(*(x.denominator for x in row))
            equalities.append([int(x * scale) for x in row])

    rays = [tuple(int(i == j) for j in range(dim)) for i in range(dim)]
    for a in equalities:
        values = [sum(x * y for x, y in zip(a, ray, strict=True)) for ray in rays]
        zeros = [sum(1 << j for j, x in enumerate(ray) if x == 0) for ray in rays]
        new = [ray for ray, value in zip(rays, values, strict=True) if value == 0]
        for p, v_p in enumerate(values):
            if v_p <= 0:
                continue
            for n, v_n in enumerate(values):
                if v_n >= 0:
                    continue
                common = zeros[p] & zeros[n]
                if any(common & zeros[k] == common for k in range(len(rays)) if k not in (p, n)):
                    continue
                ray = [v_p * x - v_n * y for x, y in zip(rays[n], rays[p], strict=True)]
                g = gcd(*ray)
                new.append(tuple(x // g for x in ray))
        rays = new
    return np.array([[x / max(ray) for x in ray] for ray in rays], dtype=float).reshape(-1, dim)


def _extreme_rays(channels, tol=1e-9):
    """
    Extreme rays of the cone C = {u >= 0 : K^(1) u_1 = K^(i) u_i for all i}.

    The rays are computed by cddlib if it is installed and its result passes verification,
    and otherwise by an exact implementation of the double description method.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    tol : float
        Tolerance for verifying the rays from cddlib.

    Returns
    -------
    rays : ndarray, shape (n_rays, sum_i |Y_i|)
        The extreme rays, each scaled to have largest entry 1.
    offsets : ndarray
        Where each u_i starts in a ray, followed by the length of a ray.
    """
    rows, offsets = _cone_equalities(channels)
    dim = offsets[-1]
    try:
        rays = _extreme_rays_cdd(rows, dim)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        rays = None
    if rays is not None and len(rays):
        scale = np.abs(rays).max(axis=1, keepdims=True)
        valid = bool((scale > 0).all())
        if valid:
            rays = rays / scale
            valid = rays.min() >= -tol and (not rows.size or np.abs(rays @ rows.T).max() <= tol)
        if not valid:
            rays = None
    if rays is None or not len(rays):
        rays = _extreme_rays_exact(channels)
    return np.clip(rays, 0.0, None), offsets


def _outcome_information(pi_t, columns):
    """
    Contribution f(k) = sum_t pi_t k_t log2(k_t / sum_t' pi_t' k_t') of an outcome of Q to
    I(Q; T), for each row k (a column of K_Q, up to scale) of `columns`.
    """
    joint = columns * pi_t
    marginal = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log2(columns / marginal), 0.0)
    return terms.sum(axis=1)


def _degradation_lp(channels, pi_t):
    """
    Compute I_d^∩ exactly, by a linear program over the extreme rays of the cone of
    common degradations.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).

    Returns
    -------
    mi : float
        The degradation intersection information (in bits).
    lambdas : list of ndarray
        The degradation channels Λ_i of an optimal Q, each shape (|Y_i|, |Q|), so that
        K_Q = K^(i) @ Λ_i for every i.
    """
    rays, offsets = _extreme_rays(channels)
    columns = rays[:, offsets[0] : offsets[1]] @ channels[0].T
    values = _outcome_information(pi_t, columns)
    res = linprog(-values, A_eq=rays.T, b_eq=np.ones(offsets[-1]), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"The linear program for I_d^∩ failed: {res.message}")
    used = res.x > 1e-12
    weighted = res.x[used, None] * rays[used]
    lambdas = [weighted[:, offsets[i] : offsets[i + 1]].T for i in range(len(channels))]
    return max(0.0, -res.fun), lambdas


def _degradation_channel(channels, pi_t, bound=None, niter=None, steps=50, seed=0):
    """
    Maximize I(Q; T) over common degradations K_Q = K^(i) @ Λ_i of every source.

    The feasible (Λ_1, ..., Λ_n) form a polytope and I(Q; T) is convex in K_Q, so
    the maximum is at a vertex. Each start is a vertex from a linear program with a
    random cost, followed by successive linearization: move to the vertex maximizing
    the gradient's linear form while that improves I(Q; T). Every candidate is
    exactly feasible.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).
    bound : int or None
        Cardinality of Q.  Defaults to Kolchinsky's bound.
    niter : int or None
        Number of random vertices to start from.
    steps : int
        Maximum linearization steps per start.
    seed : int or None
        Random seed for the vertex costs.

    Returns
    -------
    mi : float
        The degradation intersection information (in bits).
    k_q : ndarray or None
        An optimal channel K_Q, or None if the optimum is the constant channel.
    """
    n_t = channels[0].shape[0]
    sizes = [ch.shape[1] for ch in channels]
    n_q = bound if bound is not None else sum(sizes) - len(channels) + 1
    niter = 100 if niter is None else niter
    offsets = np.cumsum([0] + [s * n_q for s in sizes])
    n_z = offsets[-1]

    def kq_map(i):
        """Linear map from the stacked Λ's to vec(K^(i) @ Λ_i)."""
        M = np.zeros((n_t * n_q, n_z))
        block = np.kron(channels[i], np.eye(n_q))
        M[:, offsets[i] : offsets[i + 1]] = block
        return M

    rows = [np.kron(np.eye(s), np.ones(n_q)) for s in sizes]
    A_rows = np.zeros((sum(sizes), n_z))
    r = 0
    for i, block in enumerate(rows):
        A_rows[r : r + sizes[i], offsets[i] : offsets[i + 1]] = block
        r += sizes[i]
    M0 = kq_map(0)
    A = np.vstack([A_rows] + [M0 - kq_map(i) for i in range(1, len(channels))])
    b = np.r_[np.ones(sum(sizes)), np.zeros(A.shape[0] - sum(sizes))]

    def channel(z):
        return np.clip((M0 @ z).reshape(n_t, n_q), 0, None)

    rng = np.random.default_rng(seed)
    best_mi, best_k = 0.0, None
    for _ in range(niter):
        res = linprog(rng.standard_normal(n_z), A_eq=A, b_eq=b, bounds=(0, None), method="highs-ds")
        if res.x is None:
            continue
        z = res.x
        val = _mi_bits(pi_t, channel(z))
        for _ in range(steps):
            g = M0.T @ _mi_grad(pi_t, channel(z)).ravel()
            res = linprog(-g, A_eq=A, b_eq=b, bounds=(0, None), method="highs-ds")
            if res.x is None or _mi_bits(pi_t, channel(res.x)) <= val + 1e-12:
                break
            z, val = res.x, _mi_bits(pi_t, channel(res.x))
        if val > best_mi:
            best_mi, best_k = val, channel(z)
    return best_mi, best_k


def _degradation_ii(channels, pi_t, bound=None, niter=None):
    """
    Compute a lower bound on I_d^∩ by the local search of `_degradation_channel`.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).
    bound : int or None
        Cardinality of Q.  Defaults to Kolchinsky's bound.
    niter : int or None
        Number of random vertices to start from.

    Returns
    -------
    float
        The degradation intersection information (in bits).
    """
    return _degradation_channel(channels, pi_t, bound=bound, niter=niter)[0]


class PID_Deg(BaseBivariatePID):
    """
    Degradation intersection information I_d^∩ from Kolchinsky (2022).

    Redundancy is the maximum I(Q; T) such that Q is a Blackwell
    degradation of every source channel.  Satisfies the Williams-Beer
    axioms and the independent identity property (IIP).

    By default the maximum is computed exactly, by a linear program over
    the extreme rays of the cone of common degradations (a formulation due
    to A. F. C. Gomes). The rays are enumerated by cddlib (from the
    ``optional`` extra) when it is installed, and otherwise by an exact
    implementation of the double description method. Passing
    ``method='local'`` instead runs a local search over the vertices of the
    polytope of degradation channels, which only gives a lower bound.

    References
    ----------
    A. Kolchinsky, "A Novel Approach to the Partial Information
    Decomposition", Entropy 24, 403, 2022.
    """

    _name = "I_d∩"

    @staticmethod
    def _measure(d, sources, target, bound=None, niter=None, method=None):
        """
        Compute the degradation II redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.
        bound : int or None
            Cardinality of the auxiliary variable Q, for the local search.
        niter : int or None
            Number of random vertices to start the local search from.
        method : {'exact', 'local'} or None
            'exact' solves a linear program over the extreme rays of the cone
            of common degradations. 'local' runs a local search over the
            vertices of the polytope of degradation channels, which only gives
            a lower bound. Defaults to 'local' if `bound` or `niter` is given
            and to 'exact' otherwise.

        Returns
        -------
        float
            The degradation II redundancy (in bits).
        """
        if method is None:
            method = "exact" if bound is None and niter is None else "local"
        if method not in ("exact", "local"):
            raise ValueError(f"method must be 'exact' or 'local', not {method!r}")

        source_a, source_b = sources

        d_coal = d.coalesce([list(source_a), list(source_b), list(target)])
        src_a, src_b, tgt = d_coal.dims

        kappa_a, kappa_b, pi_t = channels_from_joint(
            d_coal,
            [tgt],
            [src_a],
            [src_b],
        )

        if method == "exact":
            return _degradation_lp([kappa_a, kappa_b], pi_t)[0]

        return _degradation_ii(
            [kappa_a, kappa_b],
            pi_t,
            bound=bound,
            niter=niter,
        )
