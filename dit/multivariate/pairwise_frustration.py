"""
Pairwise frustration: the part of a distribution's pairwise structure that cannot be
made consistent around loops, measured as the divergence from its pairwise maximum
entropy distribution to the nearest unfrustrated pairwise model.
"""

import heapq
import itertools
import math
import warnings

import networkx as nx
import numpy as np
from scipy.optimize import linprog, minimize

from ..distconst import _joint_array
from ..helpers import normalize_rvs
from ..utils import unitful

__all__ = ("pairwise_frustration",)

_LN2 = np.log(2)
_CAP = 40.0


def _entropy(p):
    m = p[p > 0]
    return float(-(m * np.log2(m)).sum())


def _marginal(p, axes):
    drop = tuple(a for a in range(p.ndim) if a not in axes)
    return p.sum(axis=drop) if drop else p


def _cmi(p, i, j, k):
    """I[X_i : X_j | X_k] in bits."""
    H = lambda axes: _entropy(_marginal(p, tuple(sorted(axes))))
    return H((i, k)) + H((j, k)) - H((k,)) - H((i, j, k))


def _kl(p, q):
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


class _Pairwise:
    """
    The pairwise exponential family on a product alphabet:
    log q = sum_i h_i(x_i) + sum_{i<j} T_ij(x_i, x_j) - log Z,
    gauge-fixed by h_i(0) = 0 and T_ij(0, .) = T_ij(., 0) = 0.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.n = len(shape)
        self.edges = list(itertools.combinations(range(self.n), 2))
        self.off_h, o = [], 0
        for k in shape:
            self.off_h.append(o)
            o += k - 1
        self.off_T = {}
        for i, j in self.edges:
            self.off_T[i, j] = o
            o += self.size(i, j)
        self.dim = o

    def size(self, i, j):
        return (self.shape[i] - 1) * (self.shape[j] - 1)

    def logq(self, v):
        E = np.zeros(self.shape)
        for i, k in enumerate(self.shape):
            sh = [1] * self.n
            sh[i] = k
            E = E + np.concatenate([[0.0], v[self.off_h[i] : self.off_h[i] + k - 1]]).reshape(sh)
        for i, j in self.edges:
            ki, kj = self.shape[i], self.shape[j]
            T = np.zeros((ki, kj))
            T[1:, 1:] = v[self.off_T[i, j] : self.off_T[i, j] + self.size(i, j)].reshape(ki - 1, kj - 1)
            sh = [1] * self.n
            sh[i], sh[j] = ki, kj
            E = E + T.reshape(sh)
        m = E.max()
        return E - (m + np.log(np.exp(E - m).sum()))

    def xent(self, v, p):
        """Cross entropy -sum p log q (nats) and its gradient."""
        lq = self.logq(v)
        q = np.exp(lq)
        g = np.zeros(self.dim)
        for i in range(self.n):
            g[self.off_h[i] : self.off_h[i] + self.shape[i] - 1] = (_marginal(q, (i,)) - _marginal(p, (i,)))[1:]
        for i, j in self.edges:
            G = (_marginal(q, (i, j)) - _marginal(p, (i, j)))[1:, 1:].ravel()
            g[self.off_T[i, j] : self.off_T[i, j] + G.size] = G
        return -(p * lq).sum(), g

    def fit(self, target, keep_edges=None, cap=_CAP):
        """
        ML fit to `target` (convex); couplings off `keep_edges` are fixed at zero.
        Parameters are bounded by `cap` so hard-core targets give finite fits.
        """
        keep = set(self.edges if keep_edges is None else keep_edges)
        bounds = [(-cap, cap)] * self.dim
        for (i, j), o in self.off_T.items():
            if (i, j) not in keep:
                bounds[o : o + self.size(i, j)] = [(0.0, 0.0)] * self.size(i, j)
        r = minimize(
            self.xent,
            np.zeros(self.dim),
            args=(target,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
        )
        return r.x

    def log_odds_row(self, i, j, A, B):
        """Coefficients of lambda_ij(A, B) = T(a1, b1) + T(a0, b0) - T(a1, b0) - T(a0, b1)."""
        if i > j:
            i, j, A, B = j, i, B, A
        row = np.zeros(self.dim)
        kj = self.shape[j]
        for (a, b), s in [((A[1], B[1]), 1), ((A[0], B[0]), 1), ((A[1], B[0]), -1), ((A[0], B[1]), -1)]:
            if a > 0 and b > 0:
                row[self.off_T[i, j] + (a - 1) * (kj - 1) + (b - 1)] += s
        return row

    def cycle_constraints(self):
        """
        For every cycle of variables and every binary sub-model (two letters per
        variable), the log-odds-ratio rows whose product must be non-negative.
        Grouped by cycle length.
        """
        seen, cycles = set(), []
        for c in nx.simple_cycles(nx.complete_graph(self.n).to_directed()):
            if len(c) < 3:
                continue
            key = frozenset(frozenset((c[t], c[(t + 1) % len(c)])) for t in range(len(c)))
            if key not in seen:
                seen.add(key)
                cycles.append(c)
        pairs = [list(itertools.combinations(range(k), 2)) for k in self.shape]
        groups = {}
        for c in cycles:
            L = len(c)
            for choice in itertools.product(*[pairs[v] for v in c]):
                rows = np.array(
                    [self.log_odds_row(c[t], c[(t + 1) % L], choice[t], choice[(t + 1) % L]) for t in range(L)]
                )
                groups.setdefault(L, []).append(rows)
        return {L: np.array(g) for L, g in groups.items()}


def _maxent_support(p):
    """
    Cells that some distribution with p's pairwise marginals puts mass on: the support of
    the pairwise maximum entropy distribution. One linear program per cell.
    """
    if (p > 0).all():
        return np.ones(p.shape, dtype=bool)
    cells = list(itertools.product(*[range(k) for k in p.shape]))
    A, b = [], []
    for i, j in itertools.combinations(range(p.ndim), 2):
        m = _marginal(p, (i, j))
        for a, c in itertools.product(range(p.shape[i]), range(p.shape[j])):
            A.append([1.0 if (x[i], x[j]) == (a, c) else 0.0 for x in cells])
            b.append(m[a, c])
    A, b = np.array(A), np.array(b)
    support = np.zeros(len(cells), dtype=bool)
    for t in range(len(cells)):
        if p.flat[t] > 0:
            support[t] = True
            continue
        c = np.zeros(len(cells))
        c[t] = -1.0
        r = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        support[t] = r.status == 0 and -r.fun > 1e-12
    return support.reshape(p.shape)


def _pairwise_maxent(p):
    """
    Max-entropy distribution with p's pairwise marginals: the ML fit of the pairwise
    family restricted to its support, where the fit is finite.
    """
    fam = _Pairwise(p.shape)
    support = _maxent_support(p)
    if support.all():
        return np.exp(fam.logq(fam.fit(p, cap=np.inf)))

    def xent(v):
        E = np.where(support, fam.logq(v), -np.inf)
        m = E[support].max()
        lq = E - (m + np.log(np.exp(E[support] - m).sum()))
        q = np.exp(lq)
        g = np.zeros(fam.dim)
        for i in range(fam.n):
            g[fam.off_h[i] : fam.off_h[i] + fam.shape[i] - 1] = (_marginal(q, (i,)) - _marginal(p, (i,)))[1:]
        for i, j in fam.edges:
            G = (_marginal(q, (i, j)) - _marginal(p, (i, j)))[1:, 1:].ravel()
            g[fam.off_T[i, j] : fam.off_T[i, j] + G.size] = G
        return -(p[support] * lq[support]).sum(), g, q

    r = minimize(
        lambda v: xent(v)[:2],
        np.zeros(fam.dim),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-16, "gtol": 1e-12},
    )
    return xent(r.x)[2]


def _binary_triangle(p2):
    """Closed form for three bits: the smallest CMI if the coupling loop is negative, else 0."""
    fam = _Pairwise((2, 2, 2))
    v = fam.fit(p2)
    loop = np.prod([v[fam.off_T[e]] for e in fam.edges])
    if loop >= 0:
        return 0.0
    return min(_cmi(p2, i, j, k) for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)])


def _tree_projection(p, T):
    lq = np.zeros(p.shape)
    deg = dict(T.degree())
    for i, j in T.edges():
        i, j = sorted((i, j))
        sh = [1] * p.ndim
        sh[i], sh[j] = p.shape[i], p.shape[j]
        lq = lq + np.log(np.maximum(_marginal(p, (i, j)), 1e-300)).reshape(sh)
    for i in range(p.ndim):
        sh = [1] * p.ndim
        sh[i] = p.shape[i]
        lq = lq - (deg.get(i, 0) - 1) * np.log(np.maximum(_marginal(p, (i,)), 1e-300)).reshape(sh)
    q = np.exp(lq)
    return q / q.sum()


def _upper_bound(p2):
    """Every tree is unfrustrated: the distance to the nearest spanning-tree model."""
    return min(_kl(p2, _tree_projection(p2, T)) for T in nx.SpanningTreeIterator(nx.complete_graph(p2.ndim)))


def _lower_bound(p2):
    """Chain rule over disjoint binary sub-boxes, each scored by the binary closed form."""
    if p2.ndim != 3:
        return 0.0
    pairs = [list(itertools.combinations(range(k), 2)) for k in p2.shape]
    boxes = []
    for S in itertools.product(*pairs):
        sub = p2[np.ix_(*S)]
        w = sub.sum()
        if w <= 0:
            continue
        f = _binary_triangle(sub / w)
        if f > 0:
            boxes.append((w * f, set(itertools.product(*S))))
    boxes.sort(key=lambda t: -t[0])
    used, total = [], 0.0
    for val, cells in boxes:
        if all(cells.isdisjoint(u) for u in used):
            used.append(cells)
            total += val
    return total


def _mtp2_project(fam, p, perm):
    """Projection onto pairwise models that are MTP2 in the orderings `perm` (convex)."""
    pp = p[np.ix_(*perm)]

    def unpack(v):
        u = v.copy()
        for (i, j), o in fam.off_T.items():
            ki, kj = fam.shape[i], fam.shape[j]
            w = v[o : o + fam.size(i, j)].reshape(ki - 1, kj - 1)
            u[o : o + fam.size(i, j)] = w.cumsum(0).cumsum(1).ravel()
        return u

    def objective(v):
        val, g = fam.xent(unpack(v), pp)
        for (i, j), o in fam.off_T.items():
            ki, kj = fam.shape[i], fam.shape[j]
            G = g[o : o + fam.size(i, j)].reshape(ki - 1, kj - 1)
            g[o : o + fam.size(i, j)] = G[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1].ravel()
        return val, g

    # in a fixed ordering, MTP2 <=> every adjacent mixed difference w of T_ij is >= 0
    bounds = [(-_CAP, _CAP)] * fam.dim
    for (i, j), o in fam.off_T.items():
        bounds[o : o + fam.size(i, j)] = [(0.0, _CAP)] * fam.size(i, j)
    r = minimize(
        objective,
        np.zeros(fam.dim),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
    )
    q = np.exp(fam.logq(unpack(r.x)))
    inv = [np.argsort(pi) for pi in perm]
    return r.fun, q[np.ix_(*inv)]


def _orderings(shape):
    per = [list(itertools.permutations(range(k))) for k in shape]
    for combo in itertools.product(*per):
        if combo[0][0] < combo[0][-1]:
            yield [np.array(c) for c in combo]


def _mtp2_start(fam, p2, max_exhaustive=500, sweeps=4):
    """Best MTP2-ordered projection: exhaustive for small alphabets, else seriation plus swaps."""
    if math.prod(math.factorial(k) for k in p2.shape) / 2 <= max_exhaustive:
        return min((_mtp2_project(fam, p2, perm) for perm in _orderings(p2.shape)), key=lambda t: t[0])[1]
    v = fam.fit(p2)
    full = {}
    for (i, j), o in fam.off_T.items():
        ki, kj = fam.shape[i], fam.shape[j]
        M = np.zeros((ki, kj))
        M[1:, 1:] = v[o : o + fam.size(i, j)].reshape(ki - 1, kj - 1)
        M = M - M.mean(0) - M.mean(1)[:, None] + M.mean()
        full[i, j], full[j, i] = M, M.T
    base = [
        np.argsort(np.linalg.svd(np.hstack([full[i, j] for j in range(fam.n) if j != i]))[0][:, 0])
        for i in range(fam.n)
    ]
    cache = {}

    def score(perm):
        key = tuple(map(tuple, perm))
        if key not in cache:
            cache[key] = _mtp2_project(fam, p2, perm)
        return cache[key]

    best = (np.inf, None)
    for signs in itertools.product([False, True], repeat=fam.n - 1):
        perm = [base[0].copy()] + [b[::-1].copy() if s else b.copy() for b, s in zip(base[1:], signs, strict=True)]
        cur = score(perm)[0]
        for _ in range(sweeps):
            improved = False
            for i in range(fam.n):
                for a in range(fam.shape[i] - 1):
                    cand = [x.copy() for x in perm]
                    cand[i][[a, a + 1]] = cand[i][[a + 1, a]]
                    d = score(cand)[0]
                    if d < cur - 1e-12:
                        perm, cur, improved = cand, d, True
            if not improved:
                break
        if cur < best[0]:
            best = (cur, score(perm)[1])
    return best[1]


def _squashed_products(groups):
    """Cycle constraints as smooth functions: prod of lambda / sqrt(1 + lambda^2) >= 0."""

    def g(v):
        return np.concatenate([np.prod(_squash(L @ v), axis=1) for L in groups.values()])

    def jac(v):
        out = []
        for L in groups.values():
            vals = L @ v
            s = _squash(vals)
            J = np.zeros((len(L), L.shape[2]))
            for t in range(L.shape[1]):
                others = np.prod(np.delete(s, t, axis=1), axis=1)
                J += (others * (1 + vals[:, t] ** 2) ** -1.5)[:, None] * L[:, t, :]
            out.append(J)
        return np.vstack(out)

    return g, jac


def _squash(x):
    return x / np.sqrt(1 + x * x)


def _local_search(p2):
    """Feasible upper bound from SLSQP over U_box, started at trees, MTP2 orderings and p2."""
    fam = _Pairwise(p2.shape)
    g, jac = _squashed_products(fam.cycle_constraints())
    starts = [fam.fit(p2)]
    starts += [
        fam.fit(p2, [tuple(sorted(e)) for e in T.edges()]) for T in nx.SpanningTreeIterator(nx.complete_graph(fam.n))
    ]
    starts.append(fam.fit(_mtp2_start(fam, p2)))
    best = np.inf
    for v0 in starts:
        r = minimize(
            fam.xent,
            v0,
            args=(p2,),
            jac=True,
            method="SLSQP",
            bounds=[(-_CAP, _CAP)] * fam.dim,
            constraints=[{"type": "ineq", "fun": g, "jac": jac}],
            options={"maxiter": 500, "ftol": 1e-13},
        )
        for v in (v0, np.clip(r.x, -_CAP, _CAP)):
            if g(v).min() >= -1e-7:
                best = min(best, fam.xent(v, p2)[0])
    return best / _LN2 - _entropy(p2)


def _branch_and_bound(p2, incumbent, max_nodes, tol):
    """
    Global minimum over U_box. Every log-odds ratio is linear in the natural parameters,
    so U_box is a finite union of polyhedral cones indexed by sign patterns. A node fixes
    the sign (>= 0, <= 0, or = 0) of some log-odds ratios; its relaxation drops the
    undetermined cycle products and is a convex projection, hence a lower bound.

    Returns (best value, certified lower bound) in bits.
    """
    fam = _Pairwise(p2.shape)
    rows, index, cons = [], {}, []
    for group in fam.cycle_constraints().values():
        for c in group:
            ids = []
            for r in c:
                key, neg = tuple(np.round(r, 12)), tuple(np.round(-r, 12))
                if key in index:
                    ids.append((index[key], 1))
                elif neg in index:
                    ids.append((index[neg], -1))
                else:
                    index[key] = len(rows)
                    rows.append(r)
                    ids.append((index[key], 1))
            cons.append(ids)
    F = np.array(rows)
    H = _entropy(p2)
    nodes = [0]

    def relax(assign, v0):
        constraints = []
        ineq = [(f, s) for f, s in assign.items() if s != 0]
        eq = [f for f, s in assign.items() if s == 0]
        if ineq:
            A = np.array([s * F[f] for f, s in ineq])
            constraints.append({"type": "ineq", "fun": lambda v, A=A: A @ v, "jac": lambda v, A=A: A})
        if eq:
            B = F[eq]
            constraints.append({"type": "eq", "fun": lambda v, B=B: B @ v, "jac": lambda v, B=B: B})
        r = minimize(
            fam.xent,
            v0,
            args=(p2,),
            jac=True,
            method="SLSQP",
            bounds=[(-_CAP, _CAP)] * fam.dim,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        nodes[0] += 1
        return r.fun / _LN2 - H, r.x

    def most_violated(v):
        vals = F @ v
        s = _squash(vals)
        worst, arg = -1e-9, None
        for c in cons:
            prod = np.prod([o * s[f] for f, o in c])
            if prod < worst:
                worst, arg = prod, c
        return arg, vals

    best = incumbent
    lb0, v = relax({}, np.zeros(fam.dim))
    heap, tick = [(lb0, 0, {}, v)], 1
    while heap and nodes[0] < max_nodes:
        lb, _, assign, v = heapq.heappop(heap)
        if lb >= best - tol:
            continue
        c, vals = most_violated(v)
        if c is None:
            best = lb
            continue
        free = [f for f, _ in c if f not in assign]
        if free:
            f = min(free, key=lambda t: abs(vals[t]))
            children = [{**assign, f: 1}, {**assign, f: -1}]
        else:
            # every sign in this cycle is fixed and their product is negative: one must vanish
            children = [{**assign, f: 0} for f, _ in c if assign[f] != 0]
        for child in children:
            val, vv = relax(child, v)
            if val < best - tol:
                heapq.heappush(heap, (val, tick, child, vv))
                tick += 1
    lower = min([best] + [h[0] for h in heap if h[0] < best])
    return best, lower


def _frustration(p, max_nodes, tol):
    """Frustration of a dense joint pmf (one axis per variable), in bits."""
    # constant variables carry no couplings
    p = p.reshape([k for k in p.shape if k > 1]) / p.sum()
    if p.ndim < 3:
        return 0.0
    p2 = _pairwise_maxent(p)
    if p2.shape == (2, 2, 2):
        return max(_binary_triangle(p2), 0.0)
    lower, upper = _lower_bound(p2), _upper_bound(p2)
    if upper - lower < tol:
        return max(upper, 0.0)
    incumbent = min(max(_local_search(p2), lower), upper)
    value, certified = _branch_and_bound(p2, incumbent, max_nodes=max_nodes, tol=tol)
    certified = max(certified, lower)
    if value - certified > max(10 * tol, 1e-6):
        msg = (
            f"pairwise_frustration did not certify optimality within max_nodes={max_nodes}; "
            f"the true value lies in [{certified:.6g}, {value:.6g}] bits."
        )
        warnings.warn(msg, stacklevel=3)
    return max(value, 0.0)


@unitful
def pairwise_frustration(dist, rvs=None, max_nodes=20000, tol=1e-7):
    r"""
    Compute the pairwise frustration: the divergence from the pairwise maximum entropy
    distribution to the nearest unfrustrated pairwise model.

    Let :math:`\tilde{p}_2` be the maximum entropy distribution with the same pairwise
    marginals as :math:`p`. Its co-information splits the co-information of :math:`p`
    into a pairwise part and the third-order connected information
    :math:`\Delta_3 = H[\tilde{p}_2] - H[p] \geq 0` :cite:`Schneidman2003`:

    .. math::
        I[X : Y : Z]_p = I[X : Y : Z]_{\tilde{p}_2} - \Delta_3 .

    The pairwise part can itself be negative, as for the W distribution, whose
    :math:`\Delta_3` is zero. Frustration measures this source of synergy through the
    couplings of :math:`\tilde{p}_2` rather than through the sign of its co-information.

    A *binary sub-model* of a pairwise model picks two letters for each variable; its
    couplings are the conditional log-odds ratios
    :math:`\lambda_{ij}(A, B)` of the chosen letter pairs. Let :math:`\mathcal{U}` be
    the set of pairwise models in which every binary sub-model is *balanced*: the
    product of log-odds ratios around every cycle of variables is non-negative. For
    bits this is the classical condition that the model can be relabeled to a
    ferromagnet :cite:`harary1953notion`, i.e. multivariate totally positive
    :cite:`fallat2017total`. Then

    .. math::
        F = \min_{q \in \mathcal{U}} D_{KL}(\tilde{p}_2 \| q)
          = \min_{q \in \mathcal{U}} D_{KL}(p \| q) - \Delta_3 ,

    the second form by the Pythagorean theorem for the pairwise exponential family
    :cite:`Amari2001`. Every tree-structured model lies in :math:`\mathcal{U}`, so
    :math:`F` is at most the divergence to the nearest tree. For three bits
    :math:`F = \min_k I[X_i : X_j | X_k]_{\tilde{p}_2}` when the loop of couplings is
    negative and zero otherwise, and :math:`F \geq -I[X : Y : Z]_{\tilde{p}_2}`
    :cite:`Cover2006`.

    For larger alphabets or more variables the minimum is computed exactly by
    branch-and-bound. Each log-odds ratio is linear in the natural parameters, so
    :math:`\mathcal{U}` is a finite union of polyhedral cones and the projection onto
    each is a convex program :cite:`lauritzen2021total`. A local search supplies the
    initial incumbent.

    No canonical literature source is known for this measure.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the frustration is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random variables
        forming one variable of the pairwise model. If None, each random variable is
        its own variable.
    max_nodes : int
        The maximum number of branch-and-bound nodes. If optimality is not certified
        within this budget, the best value found is returned with a warning giving the
        certified bracket.
    tol : float
        The optimality tolerance, in bits.

    Returns
    -------
    F : float
        The pairwise frustration.

    Examples
    --------
    >>> w = dit.Distribution(['001', '010', '100'], [1/3] * 3)
    >>> round(dit.multivariate.pairwise_frustration(w), 6)
    0.666667
    >>> dit.multivariate.pairwise_frustration(dit.example_dists.Xor())
    0.0
    """
    rvs, _ = normalize_rvs(dist, rvs, None)
    p = _joint_array(dist, rvs, None)[..., 0]
    return _frustration(p, max_nodes=max_nodes, tol=tol)
