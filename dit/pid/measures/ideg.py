"""
The degradation intersection information I_d^∩ from Kolchinsky (2022).

Defines redundancy as the maximum information a channel K_Q can carry
about T while remaining a Blackwell degradation of every source channel:

    I_d^∩(Y1, ..., Yn → T) = max_{Q : Q ≤_d Y_i ∀i} I(Q; T)

This is Kolchinsky's Blackwell redundancy I_≺ in the notation of Gomes &
Figueiredo (2023), so it is the same measure as `iprec`.

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
from ...shannon import mutual_information
from ..pid import BasePID

__all__ = ("PID_Deg",)


def _source_channels(d, sources, target):
    """
    The channels P(X_i | T) of two or more sources, and the target marginal P(T).

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    channels : list of ndarray
        The channel of each source, shape (|T|, |X_i|), over the target values with
        positive probability.
    pi_t : ndarray
        The probabilities of those target values.
    """
    d_coal = d.coalesce([list(source) for source in sources] + [list(target)])
    first_source, *other_sources, tgt = d_coal.dims
    pairs = [channels_from_joint(d_coal, [tgt], [first_source], [source]) for source in other_sources]
    first, _, pi_t = pairs[0]
    return [first] + [channel for _, channel, _ in pairs], pi_t


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


def _double_description(equalities, dim):
    """
    Extreme rays of {u >= 0 : a @ u = 0 for each a in `equalities`}, by the double
    description method in exact integer arithmetic.

    Starting from the unit vectors, which generate the orthant, each equality a @ u = 0
    keeps the rays on its hyperplane and adds, for each adjacent pair of rays on opposite
    sides of it, the ray where the face they span meets the hyperplane. Two rays are
    adjacent if no other ray vanishes on every coordinate on which both vanish.

    Parameters
    ----------
    equalities : list of list of int
        The rows a of the equalities.
    dim : int
        The dimension of u.

    Returns
    -------
    rays : ndarray, shape (n_rays, dim)
        The extreme rays, each scaled to have largest entry 1.
    """
    rays = _integer_rays(equalities, dim)
    return np.array([[x / max(ray) for x in ray] for ray in rays], dtype=float).reshape(-1, dim)


def _integer_rays(equalities, dim):
    """
    Extreme rays of {u >= 0 : a @ u = 0 for each a in `equalities`}, as tuples of
    integers with no common divisor; see :func:`_double_description`.
    """
    rays = [tuple(int(i == j) for j in range(dim)) for i in range(dim)]
    # Each equality that some ray violates lowers by at most one the dimension of the cone
    # plus the number of coordinates that vanish on all of it, so two adjacent rays, which
    # span a face of dimension 2, vanish together on at least dim - cuts - 2 coordinates.
    cuts = 0
    for a in equalities:
        values = [sum(x * y for x, y in zip(a, ray, strict=True)) for ray in rays]
        if not any(values):
            continue
        zeros = [sum(1 << j for j, x in enumerate(ray) if x == 0) for ray in rays]
        new = [ray for ray, value in zip(rays, values, strict=True) if value == 0]
        for p, v_p in enumerate(values):
            if v_p <= 0:
                continue
            for n, v_n in enumerate(values):
                if v_n >= 0:
                    continue
                common = zeros[p] & zeros[n]
                if common.bit_count() < dim - cuts - 2:
                    continue
                if any(common & zeros[k] == common for k in range(len(rays)) if k not in (p, n)):
                    continue
                ray = [v_p * x - v_n * y for x, y in zip(rays[n], rays[p], strict=True)]
                g = gcd(*ray)
                new.append(tuple(x // g for x in ray))
        rays = new
        cuts += 1
    return rays


def _integer_row(row):
    """Scale a row of fractions to integers."""
    scale = lcm(*(x.denominator for x in row))
    return [int(x * scale) for x in row]


def _extreme_rays_exact(channels):
    """
    Extreme rays of the cone of common degradations by the double description method, in
    exact integer arithmetic.

    The (exactly represented) floating-point rows of each channel are first normalized
    exactly, so that the constant degradation (the all-ones vector) lies in the cone.
    """
    return _double_description(*_exact_cone_equalities(channels))


def _exact_cone_equalities(channels):
    """
    The equalities of the cone of common degradations as rows of integers, after
    normalizing the rows of each channel exactly; see :func:`_extreme_rays_exact`.

    Returns
    -------
    equalities : list of list of int
        The rows a of the equalities a @ u = 0.
    dim : int
        The dimension of u.
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
            equalities.append(_integer_row(row))
    return equalities, dim


def _verified_cdd_rays(rows, dim, tol=1e-9):
    """
    Extreme rays of {u >= 0 : rows @ u = 0} from cddlib, each scaled to have largest entry
    1, or None if cddlib is not installed or its rays fail verification.
    """
    try:
        rays = _extreme_rays_cdd(rows, dim)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return None
    if rays is None or not len(rays):
        return None
    scale = np.abs(rays).max(axis=1, keepdims=True)
    if not (scale > 0).all():
        return None
    rays = rays / scale
    if rays.min() < -tol or (rows.size and np.abs(rays @ rows.T).max() > tol):
        return None
    return rays


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
    rays = _verified_cdd_rays(rows, offsets[-1], tol)
    if rays is None:
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


def _ray_program(rays, columns, pi_t):
    """
    The linear program over the extreme rays `rays`: the largest sum of their contributions
    to I(Q; T) whose weighted rays add up to the all-ones vector. Row r of `columns` is the
    column of K_Q, up to scale, of an outcome of Q given by ray r.
    """
    values = _outcome_information(pi_t, columns)
    return linprog(-values, A_eq=rays.T, b_eq=np.ones(rays.shape[1]), bounds=(0, None), method="highs")


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
    res = _ray_program(rays, rays[:, offsets[0] : offsets[1]] @ channels[0].T, pi_t)
    if res.status != 0:
        # On ill-conditioned channels the rays from cddlib can pass verification and still
        # miss some extreme rays, so that the constant degradation is not a combination of
        # them and the program is infeasible. The exact rays always contain it.
        rays = np.clip(_extreme_rays_exact(channels), 0.0, None)
        res = _ray_program(rays, rays[:, offsets[0] : offsets[1]] @ channels[0].T, pi_t)
    if res.status != 0:
        raise RuntimeError(f"The linear program for I_d^∩ failed: {res.message}")
    used = res.x > 1e-12
    weighted = res.x[used, None] * rays[used]
    lambdas = [weighted[:, offsets[i] : offsets[i + 1]].T for i in range(len(channels))]
    return max(0.0, -res.fun), lambdas


def _set_partitions(items, k):
    """The partitions of the list `items` into exactly `k` non-empty blocks."""
    if k == 1:
        yield [items]
    elif len(items) == k:
        yield [[item] for item in items]
    else:
        first, rest = items[0], items[1:]
        for partition in _set_partitions(rest, k - 1):
            yield [[first], *partition]
        for partition in _set_partitions(rest, k):
            for i in range(k):
                yield [*partition[:i], [first, *partition[i]], *partition[i + 1 :]]


def _restricted_degradation(channels, pi_t, n_q):
    """
    Compute I_d^∩ exactly when Q is restricted to `n_q` outcomes.

    Each outcome of Q gives a point of the cone of common degradations, so the degradation
    channels of Q are given by weights mu >= 0 on its extreme rays r with sum_r mu_r r = 1,
    each ray assigned to an outcome. The objective is convex in mu, so its maximum is
    attained at a vertex of {mu >= 0 : sum_r mu_r r = 1}, the feasible polytope of the
    linear program of :func:`_degradation_lp`, with the rays in its support grouped into
    the outcomes. The contribution of an outcome to I(Q; T) is sublinear, so splitting an
    outcome never lowers the objective: the rays of the support need only be partitioned
    into ``min(n_q, len(support))`` groups. A support has at most sum_i |Y_i| - n + 1 rays,
    so with that many outcomes the value is that of the linear program.

    The vertices are enumerated by the double description method in exact arithmetic, so
    unlike floating-point vertex enumeration this cannot fail or return points outside
    the polytope.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).
    n_q : int
        The number of outcomes of Q.

    Returns
    -------
    mi : float
        The largest I(Q; T) over common degradations Q with `n_q` outcomes (in bits).
    lambdas : list of ndarray
        The degradation channels Λ_i of an optimal Q, each shape (|Y_i|, n_q).
    """
    sizes = [ch.shape[1] for ch in channels]

    def pad(lambdas):
        return [np.pad(lam, ((0, 0), (0, n_q - lam.shape[1]))) for lam in lambdas]

    if n_q >= sum(sizes) - len(channels) + 1:
        mi, lambdas = _degradation_lp(channels, pi_t)
        if lambdas[0].shape[1] <= n_q:  # it is a vertex, unless rounding interfered
            return mi, pad(lambdas)
    equalities, dim = _exact_cone_equalities(channels)
    rays = _integer_rays(equalities, dim)
    scales = [max(ray) for ray in rays]
    unit_rays = np.array([[x / scale for x in ray] for ray, scale in zip(rays, scales, strict=True)])
    columns = unit_rays[:, : sizes[0]] @ channels[0].T
    # The vertices of {mu >= 0 : sum_r mu_r r = 1} are the extreme rays (mu, t) of the cone
    # {(mu, t) >= 0 : sum_r mu_r r = t 1} with t > 0, rescaled to t = 1.
    rows = [[ray[c] for ray in rays] + [-1] for c in range(dim)]
    best, best_atoms = -np.inf, np.zeros((0, dim))
    for vertex in _integer_rays(rows, len(rays) + 1):
        t = vertex[-1]
        support = [r for r in range(len(rays)) if vertex[r] > 0]
        if t == 0 or not support:
            continue
        mu = np.zeros(len(rays))
        mu[support] = [vertex[r] * scales[r] / t for r in support]
        for partition in _set_partitions(support, min(n_q, len(support))):
            value = _outcome_information(pi_t, np.array([mu[block] @ columns[block] for block in partition])).sum()
            if value > best:
                best, best_atoms = value, np.array([mu[block] @ unit_rays[block] for block in partition])
    offsets = np.cumsum([0, *sizes])
    return max(0.0, best), pad([best_atoms[:, offsets[i] : offsets[i + 1]].T for i in range(len(channels))])


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


class PID_Deg(BasePID):
    """
    Degradation intersection information I_d^∩ from Kolchinsky (2022).

    Redundancy is the maximum I(Q; T) such that Q is a Blackwell
    degradation of every source channel, for any number of sources.  This
    is the Blackwell redundancy of :class:`~dit.pid.PID_Prec`, in the
    notation of Gomes & Figueiredo.  Satisfies the Williams-Beer axioms and
    the independent identity property (IIP).

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

    A. F. C. Gomes and M. A. T. Figueiredo, "Orders between Channels and
    Implications for Partial Information Decomposition", Entropy 25, 975,
    2023.
    """

    _name = "I_d∩"

    @staticmethod
    def _measure(d, sources, target, bound=None, niter=None, method=None):
        """
        Compute the degradation II redundancy of some sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables.
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

        if len(sources) == 1:
            return mutual_information(d, sources[0], target)

        channels, pi_t = _source_channels(d, sources, target)

        if method == "exact":
            return _degradation_lp(channels, pi_t)[0]

        return _degradation_ii(channels, pi_t, bound=bound, niter=niter)
