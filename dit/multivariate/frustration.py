"""
Frustration: the part of a distribution's order-k interaction structure that cannot be
made consistent, measured as the divergence from its order-k maximum entropy
distribution to the nearest unfrustrated order-k model.
"""

import heapq
import itertools
import math
import warnings

import numpy as np
from scipy.optimize import linprog, minimize

from ..distconst import _joint_array
from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..utils import unitful

__all__ = (
    "frustration",
    "pairwise_frustration",
)

_LN2 = np.log(2)
_CAP = 40.0
_MAX_CYCLE_SPACE = 20


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


def _cumsum_all(w):
    for ax in range(w.ndim):
        w = w.cumsum(ax)
    return w


def _rcumsum_all(g):
    axes = tuple(range(g.ndim))
    return np.flip(_cumsum_all(np.flip(g, axes)), axes)


# ---------------------------------------------------------------- GF(2) circuits
def _gf2_rank(rows):
    """Rank over GF(2) of integer bitmask rows."""
    rows, rank = list(rows), 0
    for bit in range(max((r.bit_length() for r in rows), default=0)):
        pivot = next((r for r in rows if r >> bit & 1), None)
        if pivot is None:
            continue
        rows.remove(pivot)
        rows = [r ^ pivot if r >> bit & 1 else r for r in rows]
        rank += 1
    return rank


def _independent(sets_masks):
    return _gf2_rank(sets_masks) == len(sets_masks)


def _circuits(n, k):
    """
    Minimal even covers of the k-subsets of n variables: the circuits of the binary
    matroid of the incidence matrix. Every even cover is a disjoint union of circuits.
    """
    sets = list(itertools.combinations(range(n), k))
    masks = [sum(1 << i for i in S) for S in sets]
    m = len(sets)
    # null space of the incidence matrix over GF(2): reduce [masks | identity]
    rows = [(masks[j], 1 << j) for j in range(m)]
    basis, null = [], []
    for vec, tag in rows:
        for bvec, btag in basis:
            if vec ^ bvec < vec:
                vec, tag = vec ^ bvec, tag ^ btag
        if vec:
            basis.append((vec, tag))
            basis.sort(key=lambda t: -t[0])
        else:
            null.append(tag)
    if len(null) > _MAX_CYCLE_SPACE:
        msg = f"order {k} on {n} variables has a cycle space of dimension {len(null)}; too large to enumerate."
        raise ditException(msg)
    circuits = set()
    for r in range(1, len(null) + 1):
        for combo in itertools.combinations(null, r):
            c = 0
            for t in combo:
                c ^= t
            support = [j for j in range(m) if c >> j & 1]
            if _gf2_rank([masks[j] for j in support]) == len(support) - 1:
                circuits.add(tuple(support))
    return [[sets[j] for j in c] for c in sorted(circuits)]


# ---------------------------------------------------------------- log-linear family
class _LogLinear:
    """
    Log-linear models with interactions of order at most `order`:
    log q = sum_{|S| <= order} T_S(x_S) - log Z, gauge-fixed by T_S = 0 whenever some
    coordinate of x_S is the letter 0.
    """

    def __init__(self, shape, order):
        self.shape = tuple(shape)
        self.n = len(shape)
        self.order = order
        self.sets = [S for r in range(1, order + 1) for S in itertools.combinations(range(self.n), r)]
        self.top = [S for S in self.sets if len(S) == order]
        self.block, self.off, o = {}, {}, 0
        for S in self.sets:
            self.block[S] = tuple(self.shape[i] - 1 for i in S)
            self.off[S] = o
            o += math.prod(self.block[S])
        self.dim = o

    def slot(self, S):
        return slice(self.off[S], self.off[S] + math.prod(self.block[S]))

    def tensor(self, v, S):
        T = np.zeros(tuple(self.shape[i] for i in S))
        T[(slice(1, None),) * len(S)] = v[self.slot(S)].reshape(self.block[S])
        return T

    def logq(self, v):
        E = np.zeros(self.shape)
        for S in self.sets:
            sh = [1] * self.n
            for i in S:
                sh[i] = self.shape[i]
            E = E + self.tensor(v, S).reshape(sh)
        m = E.max()
        return E - (m + np.log(np.exp(E - m).sum()))

    def grad(self, q, p):
        g = np.zeros(self.dim)
        for S in self.sets:
            G = (_marginal(q, S) - _marginal(p, S))[(slice(1, None),) * len(S)]
            g[self.slot(S)] = G.ravel()
        return g

    def xent(self, v, p):
        """Cross entropy -sum p log q (nats) and its gradient."""
        lq = self.logq(v)
        return -(p * lq).sum(), self.grad(np.exp(lq), p)

    def fit(self, target, zero_sets=(), cap=_CAP):
        """ML fit to `target` (convex), with the couplings of `zero_sets` fixed at zero."""
        bounds = [(-cap, cap)] * self.dim
        for S in zero_sets:
            s = self.slot(S)
            bounds[s] = [(0.0, 0.0)] * (s.stop - s.start)
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

    def form_row(self, S, pairs):
        """
        Coefficients of the top-order interaction of the binary sub-model on letter pairs
        `pairs` (one (a0, a1) per variable of S): the k-th mixed difference of T_S.
        """
        row = np.zeros(self.dim)
        k = len(S)
        for b in itertools.product((0, 1), repeat=k):
            idx = tuple(pairs[t][b[t]] for t in range(k))
            if all(a > 0 for a in idx):
                flat = np.ravel_multi_index(tuple(a - 1 for a in idx), self.block[S])
                row[self.off[S] + flat] += (-1) ** (k - sum(b))
        return row

    def constraints(self, circuits):
        """
        For every circuit and every binary sub-model of its variables, the rows whose
        product must be non-negative. Returns (distinct rows, constraints as lists of
        (row index, orientation)). Letter pairs are always oriented a0 < a1, so equal
        sub-models give identical rows.
        """
        letter_pairs = [list(itertools.combinations(range(k), 2)) for k in self.shape]
        rows, index, cons = [], {}, []
        for C in circuits:
            V = sorted({i for S in C for i in S})
            for choice in itertools.product(*[letter_pairs[i] for i in V]):
                A = dict(zip(V, choice, strict=True))
                ids = []
                for S in C:
                    key = (S, tuple(A[i] for i in S))
                    if key not in index:
                        index[key] = len(rows)
                        rows.append(self.form_row(S, list(key[1])))
                    ids.append((index[key], 1))
                cons.append(ids)
        return np.array(rows), cons


# ---------------------------------------------------------------- maximum entropy
def _maxent_support(p, order):
    """
    Cells that some distribution with p's order-k marginals puts mass on: the support of
    the order-k maximum entropy distribution. One linear program per cell.
    """
    if (p > 0).all():
        return np.ones(p.shape, dtype=bool)
    cells = list(itertools.product(*[range(k) for k in p.shape]))
    A, b = [], []
    for S in itertools.combinations(range(p.ndim), order):
        m = _marginal(p, S)
        for a in itertools.product(*[range(p.shape[i]) for i in S]):
            A.append([1.0 if tuple(x[i] for i in S) == a else 0.0 for x in cells])
            b.append(m[a])
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


def _maxent(p, order):
    """Max-entropy distribution with p's order-k marginals: an ML fit restricted to its support."""
    fam = _LogLinear(p.shape, order)
    support = _maxent_support(p, order)
    if support.all():
        return np.exp(fam.logq(fam.fit(p, cap=np.inf)))

    def masked(v):
        E = np.where(support, fam.logq(v), -np.inf)
        m = E[support].max()
        lq = E - (m + np.log(np.exp(E[support] - m).sum()))
        q = np.exp(lq)
        return -(p[support] * lq[support]).sum(), fam.grad(q, p), q

    r = minimize(
        lambda v: masked(v)[:2],
        np.zeros(fam.dim),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-16, "gtol": 1e-12},
    )
    return masked(r.x)[2]


# ---------------------------------------------------------------- closed form and bounds
def _binary_triangle(p2):
    """Three bits, order 2: the smallest CMI if the coupling loop is negative, else 0."""
    fam = _LogLinear((2, 2, 2), 2)
    v = fam.fit(p2)
    if np.prod([v[fam.off[S]] for S in fam.top]) >= 0:
        return 0.0
    return min(_cmi(p2, i, j, k) for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)])


def _bases(fam, pk, max_enum=300):
    """Maximal circuit-free families of top-order sets: every such model is unfrustrated."""
    masks = {S: sum(1 << i for i in S) for S in fam.top}
    rank = _gf2_rank(list(masks.values()))
    if math.comb(len(fam.top), rank) <= max_enum:
        return [B for B in itertools.combinations(fam.top, rank) if _independent([masks[S] for S in B])]
    v = fam.fit(pk)
    strength = sorted(fam.top, key=lambda S: -np.abs(v[fam.slot(S)]).max())
    basis = []
    for S in strength:
        if _independent([masks[T] for T in basis] + [masks[S]]):
            basis.append(S)
    return [tuple(basis)]


def _kl(p, q):
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def _lower_bound(p2):
    """Order 2, three variables: chain rule over disjoint binary sub-boxes."""
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


# ---------------------------------------------------------------- ordered starts
def _ordered_project(fam, p, perm):
    """
    Projection onto models whose top-order adjacent mixed differences are all
    non-negative in the orderings `perm` (convex); every such model is unfrustrated.
    """
    pp = p[np.ix_(*perm)]

    def unpack(v):
        u = v.copy()
        for S in fam.top:
            u[fam.slot(S)] = _cumsum_all(v[fam.slot(S)].reshape(fam.block[S])).ravel()
        return u

    def objective(v):
        val, g = fam.xent(unpack(v), pp)
        for S in fam.top:
            g[fam.slot(S)] = _rcumsum_all(g[fam.slot(S)].reshape(fam.block[S])).ravel()
        return val, g

    bounds = [(-_CAP, _CAP)] * fam.dim
    for S in fam.top:
        s = fam.slot(S)
        bounds[s] = [(0.0, _CAP)] * (s.stop - s.start)
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


def _orderings(shape, order):
    per = [list(itertools.permutations(range(k))) for k in shape]
    for combo in itertools.product(*per):
        # reversing every alphabet negates top-order differences by (-1)^order
        if order % 2 == 1 or combo[0][0] < combo[0][-1]:
            yield [np.array(c) for c in combo]


def _ordered_start(fam, pk, max_exhaustive=500, sweeps=4):
    """Best ordered projection: exhaustive for small alphabets, else seriation plus swaps."""
    n_orders = math.prod(math.factorial(k) for k in pk.shape) / (2 if fam.order % 2 == 0 else 1)
    if n_orders <= max_exhaustive:
        return min((_ordered_project(fam, pk, perm) for perm in _orderings(pk.shape, fam.order)), key=lambda t: t[0])[1]
    v = fam.fit(pk)
    base = []
    for i in range(fam.n):
        blocks = []
        for S in fam.top:
            if i in S:
                T = fam.tensor(v, S)
                for ax in range(T.ndim):
                    T = T - T.mean(axis=ax, keepdims=True)
                blocks.append(np.moveaxis(T, S.index(i), 0).reshape(fam.shape[i], -1))
        base.append(np.argsort(np.linalg.svd(np.hstack(blocks))[0][:, 0]))
    cache = {}

    def score(perm):
        key = tuple(map(tuple, perm))
        if key not in cache:
            cache[key] = _ordered_project(fam, pk, perm)
        return cache[key]

    best = (np.inf, None)
    free = range(1, fam.n) if fam.order % 2 == 0 else range(fam.n)
    for flips in itertools.product([False, True], repeat=len(free)):
        perm = [b.copy() for b in base]
        for i, f in zip(free, flips, strict=True):
            if f:
                perm[i] = perm[i][::-1].copy()
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


# ---------------------------------------------------------------- local search and branch-and-bound
def _squash(x):
    return x / np.sqrt(1 + x * x)


def _product_constraint(F, cons):
    """Cycle constraints as one smooth function: prod of squashed log-odds ratios >= 0."""
    groups = {}
    for c in cons:
        groups.setdefault(len(c), []).append(c)
    idx = {L: np.array([[f for f, _ in c] for c in g]) for L, g in groups.items()}
    sgn = {L: np.array([[o for _, o in c] for c in g]) for L, g in groups.items()}

    def g(v):
        s = _squash(F @ v)
        return np.concatenate([np.prod(sgn[L] * s[idx[L]], axis=1) for L in idx])

    def jac(v):
        vals = F @ v
        s, ds = _squash(vals), (1 + vals**2) ** -1.5
        out = []
        for L in idx:
            S = sgn[L] * s[idx[L]]
            J = np.zeros((len(S), F.shape[1]))
            for t in range(L):
                others = np.prod(np.delete(S, t, axis=1), axis=1)
                J += (others * sgn[L][:, t] * ds[idx[L][:, t]])[:, None] * F[idx[L][:, t]]
            out.append(J)
        return np.vstack(out)

    return g, jac


def _local_search(fam, pk, F, cons, basis_fits):
    """Feasible upper bound from SLSQP over U, started at basis fits, the ordered start, and p_k."""
    g, jac = _product_constraint(F, cons)
    starts = [fam.fit(pk), *basis_fits, fam.fit(_ordered_start(fam, pk))]
    best = np.inf
    for v0 in starts:
        r = minimize(
            fam.xent,
            v0,
            args=(pk,),
            jac=True,
            method="SLSQP",
            bounds=[(-_CAP, _CAP)] * fam.dim,
            constraints=[{"type": "ineq", "fun": g, "jac": jac}],
            options={"maxiter": 500, "ftol": 1e-13},
        )
        for v in (v0, np.clip(r.x, -_CAP, _CAP)):
            if g(v).min() >= -1e-7:
                best = min(best, fam.xent(v, pk)[0])
    return best / _LN2 - _entropy(pk)


def _branch_and_bound(fam, pk, F, cons, incumbent, max_nodes, tol):
    """
    Global minimum over U. Every top-order interaction is linear in the parameters, so U
    is a finite union of polyhedral cones indexed by sign patterns. A node fixes the sign
    (>= 0, <= 0, or = 0) of some interactions; its relaxation drops the undetermined
    cycle products and is a convex projection, hence a lower bound.

    Returns (best value, certified lower bound) in bits.
    """
    H = _entropy(pk)
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
            args=(pk,),
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


def _frustration(p, order, max_nodes, tol):
    """Order-k frustration of a dense joint pmf (one axis per variable), in bits."""
    # constant variables carry no interactions
    p = p.reshape([k for k in p.shape if k > 1]) / p.sum()
    n = p.ndim
    if order >= n:
        return 0.0
    circuits = _circuits(n, order)
    if not circuits:
        return 0.0
    pk = _maxent(p, order)
    if order == 2 and pk.shape == (2, 2, 2):
        return max(_binary_triangle(pk), 0.0)

    fam = _LogLinear(pk.shape, order)
    fits = []
    for B in _bases(fam, pk):
        v = fam.fit(pk, [S for S in fam.top if S not in B])
        fits.append((_kl(pk, np.exp(fam.logq(v))), v))
    fits.sort(key=lambda t: t[0])
    upper = fits[0][0]
    lower = _lower_bound(pk) if order == 2 else 0.0
    if upper - lower < tol:
        return max(upper, 0.0)
    F, cons = fam.constraints(circuits)
    local = _local_search(fam, pk, F, cons, [v for _, v in fits[:5]])
    incumbent = min(max(local, lower), upper)
    value, certified = _branch_and_bound(fam, pk, F, cons, incumbent, max_nodes=max_nodes, tol=tol)
    certified = max(certified, lower)
    if value - certified > max(10 * tol, 1e-6):
        msg = (
            f"frustration did not certify optimality within max_nodes={max_nodes}; "
            f"the true value lies in [{certified:.6g}, {value:.6g}] bits."
        )
        warnings.warn(msg, stacklevel=3)
    return max(value, 0.0)


@unitful
def frustration(dist, rvs=None, order=2, max_nodes=20000, tol=1e-7):
    r"""
    Compute the order-k frustration: the divergence from the order-k maximum entropy
    distribution to the nearest model whose k-body interactions are unfrustrated.

    Let :math:`\tilde{p}_k` be the maximum entropy distribution with the same
    k-variable marginals as :math:`p`, and :math:`\Delta_j` the connected informations
    :cite:`Schneidman2003`, so that :math:`D_{KL}(p \| \tilde{p}_k) = \sum_{j > k} \Delta_j`.
    Interactions of order k can themselves be inconsistent, as the pairwise exclusions of
    the W distribution are; frustration measures this through the k-body interactions of
    :math:`\tilde{p}_k` rather than through the sign of a co-information.

    A *binary sub-model* picks two letters per variable. Its top-order interaction on a
    k-set S is the k-th mixed difference of :math:`\log q` over the chosen letters, which
    relabeling a variable in S negates. A product of such interactions over a family of
    k-sets is invariant exactly when the family covers every variable an even number of
    times (a cycle over GF(2)). Let :math:`\mathcal{U}_k` be the models of order at most
    k in which every such product, in every binary sub-model, is non-negative; lower-order
    interactions are unconstrained. For bits this is the condition that the parity
    constraints read off the signs are simultaneously satisfiable; for k = 2 it is
    balance of the signed graph of couplings :cite:`harary1953notion`, i.e. multivariate
    total positivity after relabeling :cite:`fallat2017total`. Then

    .. math::
        F_k = \min_{q \in \mathcal{U}_k} D_{KL}(\tilde{p}_k \| q)
            = \min_{q \in \mathcal{U}_k} D_{KL}(p \| q) - \sum_{j > k} \Delta_j ,

    the second form by the Pythagorean theorem for hierarchical log-linear families
    :cite:`Amari2001`. The k-subsets of k + 1 variables form a cycle only for even k, so
    for instance three-way frustration needs at least five variables. For three bits and
    k = 2, :math:`F_2 = \min_k I[X_i : X_j | X_k]_{\tilde{p}_2}` when the loop of
    couplings is negative and zero otherwise, and :math:`F_2 \geq -I[X : Y : Z]_{\tilde{p}_2}`
    :cite:`Cover2006`.

    Otherwise the minimum is computed exactly by branch-and-bound: every top-order
    interaction is linear in the natural parameters, so :math:`\mathcal{U}_k` is a finite
    union of polyhedral cones and the projection onto each is a convex program
    :cite:`lauritzen2021total`. A local search supplies the initial incumbent.

    No canonical literature source is known for this measure.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the frustration is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random variables
        forming one variable of the model. If None, each random variable is its own
        variable.
    order : int
        The interaction order k, at least 2.
    max_nodes : int
        The maximum number of branch-and-bound nodes. If optimality is not certified
        within this budget, the best value found is returned with a warning giving the
        certified bracket.
    tol : float
        The optimality tolerance, in bits.

    Returns
    -------
    F : float
        The order-k frustration.

    Raises
    ------
    ditException
        Raised if `order` is less than 2, or if the cycle space of the k-subsets is too
        large to enumerate.

    Examples
    --------
    >>> w = dit.Distribution(['001', '010', '100'], [1/3] * 3)
    >>> round(dit.multivariate.frustration(w), 6)
    0.666667
    >>> dit.multivariate.frustration(dit.example_dists.Xor())
    0.0
    """
    if order < 2:
        msg = f"order must be at least 2, {order} given."
        raise ditException(msg)
    rvs, _ = normalize_rvs(dist, rvs, None)
    p = _joint_array(dist, rvs, None)[..., 0]
    return _frustration(p, order=order, max_nodes=max_nodes, tol=tol)


@unitful
def pairwise_frustration(dist, rvs=None, max_nodes=20000, tol=1e-7):
    r"""
    Compute the pairwise frustration, the order-2 :func:`frustration`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the frustration is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random variables
        forming one variable of the pairwise model. If None, each random variable is its
        own variable.
    max_nodes : int
        The maximum number of branch-and-bound nodes.
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
    """
    rvs, _ = normalize_rvs(dist, rvs, None)
    p = _joint_array(dist, rvs, None)[..., 0]
    return _frustration(p, order=2, max_nodes=max_nodes, tol=tol)
