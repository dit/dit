"""
The extension profile of a pair of random variables and its distinguished points.

The extension profile :cite:`matveev2026beyond` of a pair :math:`(X, Y)` is the
compact convex body

.. math::

    \\mathrm{Ext}(X, Y) = \\{(H[X|W],\\ H[Y|W],\\ I[X:Y|W])\\}

over all auxiliary :math:`W`. It is an affine image of the Gray-Wyner region
:cite:`gray1974source` (which adds everything above it in rate space) and of
the mutual information region :cite:`li2018extended`, so it carries every
operating point of either: the common informations, the information
bottleneck and privacy funnel, the excess functional information, Körner
graph entropy, and the interaction informations.

`ExtensionProfile` computes the boundary of the body and all of those points
for a given pair; `plot_extension_profile` (Plotly) draws them.
"""

from functools import cached_property
from itertools import product

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.special import softmax

from ...exceptions import ditException
from ...utils.deprecation import renamed_kwarg

__all__ = (
    "ExtensionProfile",
    "excess_functional_information",
    "korner_graph_entropy",
)


def _h(p):
    """Shannon entropy in bits of a (flattened) array of probabilities."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _profile(joint):
    """(H[X|W], H[Y|W], I[X:Y|W]) of an array p(x, y, w)."""
    hw = _h(joint.sum((0, 1)))
    hxw = _h(joint.sum(1))
    hyw = _h(joint.sum(0))
    hxyw = _h(joint)
    return np.array([hxw - hw, hyw - hw, hxw + hyw - hxyw - hw])


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def _best(objective, starts):
    """Run L-BFGS-B from each start; return the best optimizer vector and its value."""
    best = None
    for x0 in starts:
        res = minimize(
            objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12}
        )
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def _merge_identical(pxy, axis):
    """Merge the rows (axis=0) or columns (axis=1) of pxy whose conditionals coincide."""
    m = pxy if axis == 0 else pxy.T
    cond = m / m.sum(1, keepdims=True)
    groups = {}
    for i, row in enumerate(np.round(cond, 12)):
        groups.setdefault(row.tobytes(), []).append(i)
    merged = np.array([m[g].sum(0) for g in groups.values()])
    return merged if axis == 0 else merged.T


def _reduce_pair(pxy):
    """Alternately merge identical p(y|x) rows and proportional p(x|y) columns until stable."""
    while True:
        reduced = _merge_identical(_merge_identical(pxy, 0), 1)
        if reduced.shape == pxy.shape:
            return reduced
        pxy = reduced


def _support_blocks(pxy):
    """The connected components of the bipartite support graph, as (rows, columns) index arrays."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nx_, ny_ = pxy.shape
    rows, cols = np.nonzero(pxy > 0)
    graph = coo_matrix((np.ones(len(rows)), (rows, nx_ + cols)), shape=(nx_ + ny_, nx_ + ny_))
    _, labels = connected_components(graph, directed=False)
    blocks = []
    for label in np.unique(labels[np.r_[rows, nx_ + cols]]):
        xs = np.flatnonzero(labels[:nx_] == label)
        ys = np.flatnonzero(labels[nx_:] == label)
        blocks.append((xs, ys))
    return blocks


def _inf_h_given_independent(pxy, max_functions):
    """inf_{W indep X} H[Y|W] by the random-function linear program (no reductions)."""
    from scipy.sparse import coo_matrix

    px = pxy.sum(1)
    nx_, ny_ = pxy.shape
    if ny_ == 1 or nx_ == 1:
        return 0.0
    if ny_**nx_ > max_functions:
        return None
    funcs = np.array(list(product(range(ny_), repeat=nx_)))
    masses = np.zeros((len(funcs), ny_))
    for x in range(nx_):
        masses[np.arange(len(funcs)), funcs[:, x]] += px[x]
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = -np.where(masses > 0, masses * np.log2(masses), 0.0).sum(1)
    rows = (np.arange(nx_) * ny_)[None, :] + funcs
    cols = np.repeat(np.arange(len(funcs)), nx_)
    a_eq = coo_matrix((np.ones(rows.size), (rows.ravel(), cols)), shape=(nx_ * ny_, len(funcs))).tocsr()
    res = linprog(cost, A_eq=a_eq, b_eq=(pxy / px[:, None]).ravel(), bounds=(0, None), method="highs")
    return res.fun


def excess_functional_information(pxy, max_functions=200_000, reduce=True):
    """
    The excess functional information :cite:`li2018strong`

    .. math::

        \\Psi(X \\to Y) = \\inf_{W \\perp X} H[Y|W] - I[X:Y],

    computed exactly. Refining any :math:`W` independent of :math:`X` by a random
    function :math:`F: \\mathcal{X} \\to \\mathcal{Y}` independent of :math:`X` only
    lowers :math:`H[Y|W]`, and :math:`H[Y | F = f] = H[f(X)]`, so the infimum is a
    linear program over distributions on functions with
    :math:`P(f(x) = y) = p(y|x)`.

    Three exact reductions shrink that program first:

    * **Common part.** With :math:`g` the Gács-Körner common part (the connected
      components of the support), a :math:`W` independent of :math:`X` is independent
      of :math:`g` and of :math:`X` given :math:`g`, and per-block auxiliaries combine
      into one, so :math:`\\inf H[Y|W] = H[g] + \\sum_k p(k) \\inf H[Y|W, g = k]`.
    * **X to its minimal sufficient statistic about Y.** Merging values with identical
      :math:`p(y|x)` leaves the infimum unchanged: an auxiliary for the merged variable
      lifts by resampling :math:`X` given its class independently of :math:`(Y, W)`.
    * **Y to its minimal sufficient statistic about X.** Merging values with
      proportional :math:`p(x|y)` also leaves it unchanged: given its class :math:`Y`
      is independent of :math:`X`, so :math:`W` can carry an independent draw of
      :math:`Y` for every class.

    Parameters
    ----------
    pxy : np.ndarray
        The joint pmf p(x, y), shape ``(|X|, |Y|)``.
    max_functions : int
        Give up (return None) if a reduced block still has more functions than this.
    reduce : bool
        Whether to apply the reductions (False solves the raw program, for testing).

    Returns
    -------
    psi : float, None
        The excess functional information.
    inf_h : float, None
        :math:`\\inf_{W \\perp X} H[Y|W] = I[X:Y] + \\Psi`, the height of the
        corner :math:`(H[X], I + \\Psi, I + \\Psi)` of the extension profile.
    """
    pxy = np.asarray(pxy, dtype=float)
    pxy = pxy[pxy.sum(1) > 0][:, pxy.sum(0) > 0]
    pxy = pxy / pxy.sum()
    mi = _h(pxy.sum(1)) + _h(pxy.sum(0)) - _h(pxy)
    if not reduce:
        inf_h = _inf_h_given_independent(pxy, max_functions)
        return (None, None) if inf_h is None else (inf_h - mi, inf_h)
    blocks = _support_blocks(pxy)
    masses = np.array([pxy[np.ix_(xs, ys)].sum() for xs, ys in blocks])
    inf_h = _h(masses)
    for (xs, ys), mass in zip(blocks, masses, strict=True):
        h = _inf_h_given_independent(_reduce_pair(pxy[np.ix_(xs, ys)] / mass), max_functions)
        if h is None:
            return None, None
        inf_h += mass * h
    return inf_h - mi, inf_h


def _prime_graph_entropy(graph, p, max_sets, iters):
    """min I[X:U] over U a maximal independent set containing X, by alternating minimization."""
    import networkx as nx

    nodes = list(graph)
    if len(nodes) == 1:
        return 0.0
    sets = []
    for clique in nx.find_cliques(nx.complement(graph)):
        sets.append(clique)
        if len(sets) > max_sets:
            return None
    index = {v: i for i, v in enumerate(nodes)}
    member = np.zeros((len(nodes), len(sets)), dtype=bool)
    for u, s in enumerate(sets):
        member[[index[v] for v in s], u] = True
    pv = np.array([p[v] for v in nodes])
    r = np.full(len(sets), 1 / len(sets))
    for _ in range(iters):
        q = np.where(member, r[None, :], 0.0)
        q /= q.sum(1, keepdims=True)
        r = pv @ q
    joint = pv[:, None] * q
    return _h(pv) + _h(joint.sum(0)) - _h(joint)


def _graph_entropy(graph, p, max_sets, iters):
    """Graph entropy of a networkx graph with vertex masses p (summing to 1), by recursive
    substitution through disjoint unions and joins, and enumeration on prime graphs."""
    import networkx as nx

    nodes = list(graph)
    if len(nodes) == 1:
        return 0.0
    for parts, joined in (
        (list(nx.connected_components(graph)), False),
        (list(nx.connected_components(nx.complement(graph))), True),
    ):
        if len(parts) > 1:
            masses = np.array([sum(p[v] for v in part) for part in parts])
            total = _h(masses) if joined else 0.0
            for part, mass in zip(parts, masses, strict=True):
                sub = graph.subgraph(part)
                total += mass * _graph_entropy(sub, {v: p[v] / mass for v in part}, max_sets, iters)
            return total
    return _prime_graph_entropy(graph, p, max_sets, iters)


def korner_graph_entropy(pxy, max_sets=5000, iters=5000, reduce=True):
    """
    Körner's graph entropy :cite:`korner1973coding` of the characteristic graph of
    :math:`X` (two values are adjacent when some :math:`y` is compatible with both),

    .. math::

        H_K(X) = \\min_{X \\in U \\in \\Gamma} I[X:U],

    with :math:`\\Gamma` the maximal independent sets. In the extension profile it fixes
    the point :math:`(x, H[Y|X] + x, x)` with :math:`x = H[X] - H_K(X)` on the edge
    :math:`U - X - Y`, :math:`H[X|Y, U] = 0` :cite:`li2018extended`.

    The graph is first reduced by substitution :cite:`korner1973coding`: values with the
    same support set are adjacent twins, so each class collapses to one vertex at a cost of
    :math:`p(\\text{class}) H[X|\\text{class}]`; a disconnected graph contributes the
    weighted entropies of its components, and a join (disconnected complement) adds the
    entropy of the part masses. Only prime graphs enumerate maximal independent sets, on
    which the convex minimization is solved by Blahut-Arimoto style alternating
    minimization.

    Parameters
    ----------
    pxy : np.ndarray
        The joint pmf p(x, y).
    max_sets : int
        Give up (return None) if a prime graph has more maximal independent sets.
    iters : int
        Alternating-minimization iterations.
    reduce : bool
        Whether to apply the reductions (False enumerates on the full graph, for testing).

    Returns
    -------
    h_k : float, None
        The graph entropy.
    """
    import networkx as nx

    pxy = np.asarray(pxy, dtype=float)
    pxy = pxy[pxy.sum(1) > 0] / pxy.sum()
    support = pxy > 0
    px = pxy.sum(1)
    if reduce:
        classes = {}
        for i, row in enumerate(support):
            classes.setdefault(row.tobytes(), []).append(i)
        groups = list(classes.values())
        masses = np.array([px[g].sum() for g in groups])
        twins = sum(m * _h(px[g] / m) for g, m in zip(groups, masses, strict=True))
        patterns = np.array([support[g[0]] for g in groups])
    else:
        masses, twins, patterns = px, 0.0, support
    graph = nx.Graph()
    graph.add_nodes_from(range(len(masses)))
    graph.add_edges_from(
        (a, b) for a in range(len(masses)) for b in range(a + 1, len(masses)) if np.any(patterns[a] & patterns[b])
    )
    p = dict(enumerate(masses))
    if not reduce:
        return _prime_graph_entropy(graph, p, max_sets, iters)
    h = _graph_entropy(graph, p, max_sets, iters)
    return None if h is None else twins + h


class ExtensionProfile:
    """
    The extension profile of a pair and its distinguished points.

    The lower boundary is sampled over every downward direction
    :math:`(\\alpha, \\beta, -1)`, written as :math:`\\alpha + i\\beta = r e^{i\\theta}`;
    for each :math:`\\theta` the optimum is continued along :math:`r` (from
    :math:`r \\to 0`, the rim of the bottom face :math:`I[X:Y|W] = 0`, to
    :math:`r \\to \\infty`, the silhouette), then every grid point is re-solved
    from its neighbours until none improves. Points with negligible
    :math:`I[X:Y|W]` are snapped to the bottom face. The upper boundary is sampled
    over upward directions. All of these are attained points, so their hull is an
    inner approximation of the body.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The two variables ``[X, Y]``. If None, the two variables of `dist`.
    bound : int, None
        Cardinality of the auxiliary W in the boundary optimizations. If None,
        ``min(support + 2, 8)``.
    rng : int, numpy.random.Generator, None
        Seed or generator for the optimizers' random starts.
    """

    @renamed_kwarg("seed", "rng")
    def __init__(self, dist, rvs=None, bound=None, rng=None):
        if rvs is None:
            if dist.outcome_length() != 2:
                msg = "ExtensionProfile needs `rvs` unless the distribution has exactly two variables."
                raise ditException(msg)
            rvs = [[0], [1]]
        if len(rvs) != 2:
            msg = f"The extension profile is defined for pairs, not {len(rvs)} variables."
            raise ditException(msg)

        d = dist.copy(base="linear").coalesce(rvs)
        d.make_sparse()
        xs = sorted({o[0] for o in d.outcomes}, key=repr)
        ys = sorted({o[1] for o in d.outcomes}, key=repr)
        self.pxy = np.zeros((len(xs), len(ys)))
        for o, p in zip(d.outcomes, d.pmf, strict=True):
            self.pxy[xs.index(o[0]), ys.index(o[1])] += p
        self.pair = d
        self._live = self.pxy > 0
        self.bound = bound if bound is not None else min(int(self._live.sum()) + 2, 8)
        self._rng = np.random.default_rng(rng)

        self.h_x = _h(self.pxy.sum(1))
        self.h_y = _h(self.pxy.sum(0))
        self.h_xy = _h(self.pxy)
        self.mutual_information = self.h_x + self.h_y - self.h_xy
        self.h_x_given_y = self.h_xy - self.h_y
        self.h_y_given_x = self.h_xy - self.h_x

    # ------------------------------------------------------------------ helpers

    def _random(self):
        return self._rng.normal(scale=3, size=int(self._live.sum()) * self.bound)

    def _support_objective(self, u):
        """-(u . (x, y, z)) and its gradient over softmax-parameterized p(w | x, y)."""
        a, b, c = u
        ca, cb, cxy, cw = a + c, b + c, -c, -(a + b + c)
        pl = self.pxy[self._live]
        k = self.bound

        def neg(theta):
            q = softmax(theta.reshape(-1, k), axis=1)
            jl = np.maximum(pl[:, None] * q, 1e-300)
            j = np.zeros(self.pxy.shape + (k,))
            j[self._live] = jl
            pxw = np.maximum(j.sum(1), 1e-300)
            pyw = np.maximum(j.sum(0), 1e-300)
            pw = np.maximum(j.sum((0, 1)), 1e-300)
            val = ca * _h(pxw) + cb * _h(pyw) + cxy * _h(jl) + cw * _h(pw)
            g = -(ca * np.log2(pxw)[:, None, :] + cb * np.log2(pyw)[None, :, :] + cw * np.log2(pw)[None, None, :])
            g = g[self._live] - cxy * np.log2(jl)
            gt = pl[:, None] * q * (g - (q * g).sum(1, keepdims=True))
            return -val, -gt.ravel()

        return neg

    def _point(self, theta):
        q = softmax(theta.reshape(-1, self.bound), axis=1)
        j = np.zeros(self.pxy.shape + (self.bound,))
        j[self._live] = self.pxy[self._live][:, None] * q
        return _profile(j)

    def _realizer(self, dist, w):
        """The profile point of the variables `w` of a dit distribution extending the pair."""
        from ...multivariate import coinformation, entropy

        return np.array([entropy(dist, [0], w), entropy(dist, [1], w), coinformation(dist, [[0], [1]], w)])

    def support_point(self, direction, starts=8):
        """
        A point of the extension profile maximizing ``direction . (x, y, z)``.

        Parameters
        ----------
        direction : array-like
            The direction (any length; it need not be normalized).
        starts : int
            Number of random starts.

        Returns
        -------
        point : np.ndarray
            The supporting point ``(H[X|W], H[Y|W], I[X:Y|W])``.
        """
        theta, _ = _best(self._support_objective(np.asarray(direction, float)), [self._random() for _ in range(starts)])
        return self._point(theta)

    # ---------------------------------------------------------------- the body

    @cached_property
    def probes(self):
        """The points of W = XY, X, Y and const, the vertices of Ext_min :cite:`matveev2026beyond`."""
        return {
            "W = XY": np.zeros(3),
            "W = X": np.array([0.0, self.h_y_given_x, 0.0]),
            "W = Y": np.array([self.h_x_given_y, 0.0, 0.0]),
            "W = const": np.array([self.h_x, self.h_y, self.mutual_information]),
        }

    @cached_property
    def ext_max(self):
        """Vertices and edges of the Shannon outer bound Ext_max :cite:`matveev2026beyond`."""
        planes = [
            ((-1, 0, 0), 0),
            ((0, -1, 0), 0),
            ((0, 0, -1), 0),
            ((1, 0, 0), self.h_x),
            ((0, 1, 0), self.h_y),
            ((-1, 0, 1), 0),
            ((0, -1, 1), 0),
            ((1, 0, -1), self.h_x_given_y),
            ((0, 1, -1), self.h_y_given_x),
        ]
        n = np.array([p[0] for p in planes], float)
        c = np.array([p[1] for p in planes], float)
        verts, tight = [], []
        for idx in product(range(len(planes)), repeat=3):
            if not idx[0] < idx[1] < idx[2]:
                continue
            a = n[list(idx)]
            if abs(np.linalg.det(a)) < 1e-9:
                continue
            v = np.linalg.solve(a, c[list(idx)])
            if np.all(n @ v <= c + 1e-9) and not any(np.allclose(v, u) for u in verts):
                verts.append(v)
                tight.append(set(np.flatnonzero(np.abs(n @ v - c) < 1e-9)))
        edges = [
            (verts[i], verts[j])
            for i in range(len(verts))
            for j in range(i + 1, len(verts))
            if len(tight[i] & tight[j]) >= 2 and np.linalg.matrix_rank(n[list(tight[i] & tight[j])]) == 2
        ]
        return np.array(verts), edges

    def lower_boundary(self, num_theta=48, num_radii=33, r_min=1e-5, r_max=1e3, max_passes=6):
        """
        Sample the lower (downward-facing) boundary.

        Parameters
        ----------
        num_theta : int
            Number of angles round the circle.
        num_radii : int
            Number of radii, geometrically spaced in ``[r_min, r_max]``.
        r_min, r_max : float
            The radial range; small radii resolve the bottom-face rim.
        max_passes : int
            Maximum neighbour-relaxation passes.

        Returns
        -------
        points : np.ndarray
            Shape ``(num_theta, num_radii, 3)``.
        """
        thetas = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
        radii = np.geomspace(r_min, r_max, num_radii)
        mid = int(np.argmin(np.abs(np.log(radii))))
        objs = [[self._support_objective((np.cos(t), np.sin(t), -1 / r)) for r in radii] for t in thetas]
        sols = [[None] * num_radii for _ in thetas]
        vals = np.zeros((num_theta, num_radii))
        for i in range(num_theta):
            sols[i][mid], vals[i, mid] = _best(objs[i][mid], [self._random() for _ in range(6)])
            for indices in (range(mid - 1, -1, -1), range(mid + 1, num_radii)):
                x = sols[i][mid]
                for k in indices:
                    x, vals[i, k] = _best(objs[i][k], [x, self._random()])
                    sols[i][k] = x
        for _ in range(max_passes):
            improved = 0
            for i in range(num_theta):
                for k in range(num_radii):
                    for ni, nk in (((i - 1) % num_theta, k), ((i + 1) % num_theta, k), (i, k - 1), (i, k + 1)):
                        if not 0 <= nk < num_radii:
                            continue
                        x, v = _best(objs[i][k], [sols[ni][nk]])
                        if v < vals[i, k] - 1e-12:
                            sols[i][k], vals[i, k] = x, v
                            improved += 1
            if not improved:
                break
        points = np.array([[self._point(x) for x in row] for row in sols])
        points[np.abs(points[..., 2]) < 1e-9, 2] = 0.0
        return points

    def upper_boundary(self, num=300, starts=4):
        """
        Sample the upper (upward-facing) boundary in `num` directions.

        Returns
        -------
        points : np.ndarray
            Shape ``(m, 3)``.
        """
        dirs = _fibonacci_sphere(2 * num)
        return np.array([self.support_point(u, starts=starts) for u in dirs if u[2] >= 0])

    # ------------------------------------------------------- common informations

    @cached_property
    def gacs_korner(self):
        """The Gács-Körner point, W = the common part :cite:`Gacs1973`."""
        from ...algorithms.lattice import insert_meet

        return self._realizer(insert_meet(self.pair, -1, [[0], [1]]), [2])

    def _optimizer_point(self, cls, niter, **kwargs):
        opt = cls(self.pair, [[0], [1]], **kwargs)
        opt.optimize(niter=niter, rng=self._rng)
        return self._realizer(opt.construct_distribution(), [2])

    def wyner(self, niter=None):
        """The Wyner point :cite:`wyner1975common`, the realizer of the Wyner common information."""
        from ...multivariate.common_informations.wyner_common_information import WynerCommonInformation

        return self._optimizer_point(WynerCommonInformation, niter)

    def exact(self, niter=None):
        """The point of a realizer of the exact common information :cite:`kumar2014exact`."""
        from ...multivariate.common_informations.exact_common_information import ExactCommonInformation

        return self._optimizer_point(ExactCommonInformation, niter)

    @cached_property
    def functional(self):
        """The functional common information point :cite:`james2017multivariate`."""
        from ...multivariate.common_informations.functional_common_information import functional_variable

        d = functional_variable(self.pair, [[0], [1]])
        return self._realizer(d, [d.outcome_length() - 1])

    @cached_property
    def mss(self):
        """The MSS common information point: W = both minimal sufficient statistics."""
        from ...algorithms.minimal_sufficient_statistic import insert_mss

        d = insert_mss(insert_mss(self.pair, -1, [0], [1]), -1, [1], [0])
        return self._realizer(d, [2, 3])

    @cached_property
    def kamath(self):
        """
        The far ends of the Kamath edges :cite:`kamath2010dual`: W = the minimal sufficient
        statistic of Y for X (on ``x = H[X|Y]``) and of X for Y. These are also the
        necessary-conditional-entropy points :cite:`cuff2010coordination`.

        Returns
        -------
        points : list of np.ndarray
            ``[Y -> X, X -> Y]``.
        values : dict
            The Kamath common informations G and necessary conditional entropies.
        """
        from ...algorithms.minimal_sufficient_statistic import insert_mss
        from ...multivariate import entropy

        out, values = [], {}
        for rv, about, tag in (([1], [0], "Y -> X"), ([0], [1], "X -> Y")):
            d = insert_mss(self.pair, -1, rv, about)
            out.append(self._realizer(d, [2]))
            values[f"G {tag}"] = entropy(d, [2])
            values[f"necessary conditional entropy {tag}"] = entropy(d, [2], about)
        return out, values

    @cached_property
    def entanglement(self):
        """
        The entanglement :cite:`makarychev2002class` point, exposed by direction (1, 1, -3):
        ``E = H[X|Y] + H[Y|X] - x - y + 3z``. Returns (point, E).
        """
        p = self.support_point((1, 1, -3), starts=12)
        return p, self.h_x_given_y + self.h_y_given_x - p[0] - p[1] + 3 * p[2]

    def relaxed_wyner(self, num=25):
        """The relaxed-Wyner curve :cite:`gastpar2019relaxed`: the diagonal alpha = beta in (0, 1/2]."""
        return np.array([self.support_point((a, a, -1), starts=6) for a in np.geomspace(1e-3, 0.5, num)])

    def beta_common(self, betas=None, niter=None):
        """Realizer points of the beta-common information :cite:`yu2017generalized` for each beta."""
        from ...divergences.maximum_correlation import maximum_correlation
        from ...multivariate.common_informations.beta_common_information import BetaCommonInformation

        rho = maximum_correlation(self.pair, [[0], [1]])
        betas = np.linspace(0, rho, 9) if betas is None else betas
        points = []
        for b in betas:
            if b >= rho - 1e-10:
                points.append(self.probes["W = const"])
            else:
                points.append(self._optimizer_point(BetaCommonInformation, niter, beta=b))
        return np.asarray(betas), np.array(points)

    # ------------------------------------- the zero-leakage faces: IB and privacy funnel

    def _face_objective(self, pxy, phi, k):
        c, s = np.cos(phi), np.sin(phi)
        ys = np.flatnonzero(pxy.sum(0) > 0)
        sub = pxy[:, ys]

        def neg(theta):
            q = softmax(theta.reshape(len(ys), k), axis=1)
            j = np.maximum(sub[:, :, None] * q[None, :, :], 1e-300)
            pxu, pyu, pu = j.sum(1), j.sum(0), j.sum((0, 1))
            val = c * (_h(pxu) - _h(pu)) + s * (_h(pyu) - _h(pu))
            g = -(c * np.log2(pxu)[:, None, :] + s * np.log2(pyu)[None, :, :] - (c + s) * np.log2(pu)[None, None, :])
            gq = (g * sub[:, :, None]).sum(0)
            return -val, -(q * (gq - (q * gq).sum(1, keepdims=True))).ravel()

        def point(theta):
            q = softmax(theta.reshape(len(ys), k), axis=1)
            return _profile(sub[:, :, None] * q[None, :, :])

        return neg, point, len(ys) * k

    def _face_arc(self, phis, swap, starts=12):
        pxy = self.pxy.T if swap else self.pxy
        k = int((pxy.sum(0) > 0).sum()) + 1
        out, x = [], None
        for phi in phis:
            neg, point, size = self._face_objective(pxy, phi, k)
            x0s = [self._rng.normal(scale=4, size=size) for _ in range(starts if x is None else 2)]
            x, _ = _best(neg, ([] if x is None else [x]) + x0s)
            p = point(x)
            out.append(p[[1, 0, 2]] if swap else p)
        return np.array(out)

    def information_bottleneck(self, num=60):
        """
        Both information-bottleneck arcs :cite:`tishby2000information,li2018extended`: the lower
        arcs of the zero-leakage faces ``x - z = H[X|Y]`` (U generated from Y) and its mirror,
        from W = const to the Kamath point.

        Returns
        -------
        arcs : list of np.ndarray
            ``[X -> Y, Y -> X]``.
        """
        phis = np.linspace(np.pi / 2 + 1e-3, np.pi - 1e-3, num)
        return [self._face_arc(phis, swap=False), self._face_arc(phis, swap=True)]

    def privacy_funnel(self, num=60):
        """
        Both privacy-funnel arcs :cite:`makhdoumi2014information,li2018extended`: the upper arcs
        of the zero-leakage faces, from W = Y (or X) to perfect privacy.

        Returns
        -------
        arcs : list of np.ndarray
            ``[X -> Y, Y -> X]``.
        """
        phis = np.linspace(-np.pi / 2 + 1e-3, -1e-3, num)
        return [self._face_arc(phis, swap=False), self._face_arc(phis, swap=True)]

    # --------------------------------------- points the Gray-Wyner region cannot see

    @cached_property
    def excess_functional_information(self):
        """
        The side corners ``(H[X], I + Psi, I + Psi)`` and mirror :cite:`li2018strong`, exact by
        linear program (None for alphabets too large). Returns (points, {direction: Psi}).
        """
        points, values = [], {}
        for pxy, tag, swap in ((self.pxy, "X -> Y", False), (self.pxy.T, "Y -> X", True)):
            psi, height = excess_functional_information(pxy)
            values[tag] = psi
            if height is not None:
                p = np.array([self.h_x, height, height]) if not swap else np.array([height, self.h_y, height])
                points.append(p)
        return points, values

    @cached_property
    def korner(self):
        """Körner graph-entropy points :cite:`korner1973coding,li2018extended` and their values."""
        points, values = [], {}
        for pxy, tag, swap in ((self.pxy, "X", False), (self.pxy.T, "Y", True)):
            h_k = korner_graph_entropy(pxy)
            values[tag] = h_k
            if h_k is not None:
                # U - X - Y and H[X|Y, U] = 0 give z = x and y = H[Y|X] + x.
                h_a, h_b_given_a = (self.h_y, self.h_x_given_y) if swap else (self.h_x, self.h_y_given_x)
                x = h_a - h_k
                p = np.array([x, h_b_given_a + x, x])
                points.append(p[[1, 0, 2]] if swap else p)
        return points, values

    @cached_property
    def interaction_information(self):
        """
        The three interaction informations of :cite:`li2018extended`, each a maximization of
        ``I[X:Y|U] - I[X:Y]``, and their points:

        * maximal, ``G_NNI``: the point of maximal ``z``;
        * asymmetric private, ``G_PNI`` (``U`` independent of ``X``): the far end
          ``(H[X], I + G, I + G)`` of the edge ``{x = H[X], z = y}`` whose near end is the
          excess-functional-information corner, and its mirror;
        * symmetric private, ``G_PPI`` (``U`` independent of ``X`` and ``Y``):
          ``(H[X], H[Y], I + G)``.

        Returns
        -------
        points : dict
            ``{"maximal": p, "asymmetric private X -> Y": p, "asymmetric private Y -> X": p,
            "symmetric private": p}``.
        values : dict
            The same keys, mapped to the values.
        """
        from .region import (
            asymmetric_private_interaction_information,
            maximal_interaction_information,
            symmetric_private_interaction_information,
        )

        i = self.mutual_information
        values = {
            "maximal": maximal_interaction_information(self.pair),
            "asymmetric private X -> Y": asymmetric_private_interaction_information(self.pair),
            "asymmetric private Y -> X": asymmetric_private_interaction_information(self.pair, rvs=[[1], [0]]),
            "symmetric private": symmetric_private_interaction_information(self.pair),
        }
        g_xy, g_yx = values["asymmetric private X -> Y"], values["asymmetric private Y -> X"]
        points = {
            "maximal": self.support_point((0, 0, 1), starts=12),
            "asymmetric private X -> Y": np.array([self.h_x, i + g_xy, i + g_xy]),
            "asymmetric private Y -> X": np.array([i + g_yx, self.h_y, i + g_yx]),
            "symmetric private": np.array([self.h_x, self.h_y, i + values["symmetric private"]]),
        }
        return points, values

    @cached_property
    def perfect_privacy(self):
        """
        The maximum rate for perfect privacy :cite:`li2018extended,rassouli2021perfect`,
        ``G_R*(X -> Y) = max I[Y:U]`` with ``X - Y - U`` and ``U`` independent of ``X``:
        the end ``(H[X], H[Y] - G, I)`` of the privacy-funnel arc, and its mirror. Computed
        exactly by `perfect_privacy_utility` (vertices of the perfect-privacy polytope).

        Returns
        -------
        points : list of np.ndarray
            ``[X -> Y, Y -> X]``.
        values : dict
            The two rates.
        """
        from ..privacy_funnel import perfect_privacy_utility

        g_xy = perfect_privacy_utility(self.pair, rvs=[[0], [1]])
        g_yx = perfect_privacy_utility(self.pair, rvs=[[1], [0]])
        i = self.mutual_information
        points = [np.array([self.h_x, self.h_y - g_xy, i]), np.array([self.h_x - g_yx, self.h_y, i])]
        return points, {"X -> Y": g_xy, "Y -> X": g_yx}

    def _bottom_rim(self, lower):
        """
        Densely sampled boundary of the bottom face :math:`I[X:Y|W] = 0`, as ``(x, y)`` pairs.

        Every rate used by channel synthesis is affine on the bottom face, so its optima
        over the (convex) face lie on this rim.
        """
        base = np.vstack([lower.reshape(-1, 3), *self.probes.values(), *self.kamath[0]])
        base = base[np.abs(base[:, 2]) < 1e-9][:, :2]
        try:
            ring = base[ConvexHull(base).vertices]
        except QhullError:  # the bottom face is a point or a segment
            ring = base
        ring = np.vstack([ring, ring[:1]])
        return np.vstack([np.linspace(a, b, 60, endpoint=False) for a, b in zip(ring[:-1], ring[1:], strict=True)])

    def public_channel_synthesis(self, lower, num=45):
        """
        The communication rate for distributed channel synthesis over a public channel
        with common-randomness rate ``t`` :cite:`cuff2013distributed`,

        .. math::

            C_{pub}(t, X \\to Y) = \\min_{X - U - Y,\\ I[XY:U] \\le t} I[X:U].

        The public message must look independent of the synthesized pair, so the common
        randomness must cover all of :math:`I[XY:U]`: the curve starts at the Wyner point,
        ``t`` equal to the Wyner common information, and reaches ``C = I[X:Y]`` at
        ``U = Y``, where ``t = H[Y]``. In the mutual information region it is the lower
        edge of the bottom face's projection onto the :math:`(I[X:U], I[XY:U])` plane.

        Parameters
        ----------
        lower : np.ndarray
            Lower-boundary points (from `lower_boundary`); their ``z = 0`` subset gives the rim.
        num : int
            Number of values of ``t``.

        Returns
        -------
        ts : np.ndarray
            The common-randomness rates.
        curves : list of (np.ndarray, np.ndarray)
            ``[(values, points) for X -> Y, (values, points) for Y -> X]``.
        """
        rim = self._bottom_rim(lower)
        v_xy = self.h_xy - rim[:, 0] - rim[:, 1]
        ts = np.linspace(v_xy.min(), max(self.h_x, self.h_y), num)
        curves = []
        for v_a in (self.h_x - rim[:, 0], self.h_y - rim[:, 1]):
            values, points = [], []
            for t in ts:
                cost = np.where(v_xy <= t + 1e-12, v_a, np.inf)
                k = int(np.argmin(cost))
                values.append(cost[k])
                points.append([rim[k, 0], rim[k, 1], 0.0])
            curves.append((np.array(values), np.array(points)))
        return ts, curves

    def channel_synthesis(self, lower, num=45):
        """
        The communication rate for distributed channel synthesis with common-randomness rate
        ``t`` :cite:`cuff2013distributed,li2018extended`,

        .. math::

            C(t, X \\to Y) = \\min_{X - U - Y} \\max\\{I[X:U],\\ I[XY:U] - t\\},

        a minimization over the bottom face of the extension profile. Its optimal ``U`` runs
        along the bottom-face rim from the Wyner point (``t = 0``) towards the Kamath point,
        where ``C(t) = I[X:Y]``. This interpolates the rim for plotting; for the rate at a
        single ``t``, optimized directly, use `dit.multivariate.channel_synthesis_rate`.

        In region-of-tension coordinates, :math:`I[X:U] = I[X:Y] + \\tau_2` and the common
        randomness used at a corner point is :math:`I[Y:U|X] = \\tau_1`, so
        :math:`C(t) = I[X:Y] + \\min_{\\tau_3 = 0} \\{\\tau_2 + (\\tau_1 - t)^+\\}`. The decoder
        needs local randomness :math:`H[Y|U]`, the extension-profile ``y`` coordinate.

        Parameters
        ----------
        lower : np.ndarray
            Lower-boundary points (from `lower_boundary`); their ``z = 0`` subset gives the rim.
        num : int
            Number of values of ``t``, from 0 to a little past the rate at which
            ``C(t) = I[X:Y]`` (the necessary conditional entropy).

        Returns
        -------
        ts : np.ndarray
            The common-randomness rates.
        curves : list of (np.ndarray, np.ndarray)
            ``[(values, points) for X -> Y, (values, points) for Y -> X]``.
        """
        rim = self._bottom_rim(lower)
        _, kv = self.kamath
        t_max = 1.1 * max(kv["necessary conditional entropy Y -> X"], kv["necessary conditional entropy X -> Y"], 1e-9)
        ts = np.linspace(0, t_max, num)
        v_xy = self.h_xy - rim[:, 0] - rim[:, 1]
        curves = []
        for v_a in (self.h_x - rim[:, 0], self.h_y - rim[:, 1]):
            values, points = [], []
            for t in ts:
                cost = np.maximum(v_a, v_xy - t)
                k = int(np.argmin(cost))
                values.append(cost[k])
                points.append([rim[k, 0], rim[k, 1], 0.0])
            curves.append((np.array(values), np.array(points)))
        return ts, curves

    # -------------------------------------------------------------- everything

    def compute(self, lower=None, upper=None, curves=None, slow=True, niter=None):
        """
        Compute the boundary and every distinguished point.

        Parameters
        ----------
        lower, upper, curves : dict, None
            Keyword arguments for `lower_boundary`, `upper_boundary` and the arc
            methods (``num``).
        slow : bool
            Whether to run the non-convex realizer optimizations (Wyner, exact,
            beta).
        niter : int, None
            Basin hops for those optimizations.

        Returns
        -------
        features : dict
            Everything `plot_extension_profile` draws.
        """
        lower, upper, curves = lower or {}, upper or {}, curves or {}
        low = self.lower_boundary(**lower)
        kamath_points, kamath_values = self.kamath
        efi_points, efi_values = self.excess_functional_information
        korner_points, korner_values = self.korner
        interaction_points, interaction_values = self.interaction_information
        perfect_points, perfect_values = self.perfect_privacy
        cs_ts, cs_curves = self.channel_synthesis(low)
        pcs_ts, pcs_curves = self.public_channel_synthesis(low)
        entanglement_point, entanglement = self.entanglement
        features = {
            "lower": low.reshape(-1, 3),
            "upper": self.upper_boundary(**upper),
            "probes": self.probes,
            "ext_max": self.ext_max,
            "Gács–Körner": self.gacs_korner,
            "functional": self.functional,
            "MSS": self.mss,
            "Kamath": kamath_points,
            "Kamath values": kamath_values,
            "entanglement": entanglement_point,
            "entanglement value": entanglement,
            "relaxed Wyner": self.relaxed_wyner(**curves),
            "information bottleneck": self.information_bottleneck(**curves),
            "privacy funnel": self.privacy_funnel(**curves),
            "excess functional information": efi_points,
            "excess functional information values": efi_values,
            "Körner graph entropy": korner_points,
            "Körner graph entropy values": korner_values,
            "interaction informations": interaction_points,
            "interaction information values": interaction_values,
            "perfect privacy": perfect_points,
            "perfect privacy values": perfect_values,
            "channel synthesis": (cs_ts, cs_curves),
            "public channel synthesis": (pcs_ts, pcs_curves),
        }
        if slow:
            features["Wyner"] = self.wyner(niter=niter)
            features["exact"] = self.exact(niter=niter)
            features["β-common"] = self.beta_common(niter=niter)
        return features

    def plot(self, features=None, **kwargs):
        """
        Draw the extension profile with Plotly. See `plot_extension_profile`.

        Parameters
        ----------
        features : dict, None
            Output of `compute`. If None, it is computed with defaults.
        kwargs : dict
            Passed to `plot_extension_profile`.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The interactive figure.
        """
        from .extension_plotting import plot_extension_profile

        return plot_extension_profile(self, self.compute() if features is None else features, **kwargs)

    def plot_mutual_information_region(self, features=None, **kwargs):
        """
        Draw the mutual information region :cite:`li2018extended` with Plotly. See
        `plot_mutual_information_region`.
        """
        from .extension_plotting import plot_mutual_information_region

        return plot_mutual_information_region(self, self.compute() if features is None else features, **kwargs)

    def plot_tension_region(self, features=None, **kwargs):
        """
        Draw the region of tension :cite:`prabhakaran2014assisted` with Plotly. See
        `plot_tension_region`.
        """
        from .extension_plotting import plot_tension_region

        return plot_tension_region(self, self.compute() if features is None else features, **kwargs)
