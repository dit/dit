"""
Amari m-flat hierarchy via reverse-KL m-projections.

The e-flat (log-linear / MaxEnt) ladder matches k-way marginals and zeros
higher-order natural parameters. Dually, the m-flat mixture hierarchy truncates
the ANOVA / Hoeffding decomposition of the *probability mass function* itself:
distributions in :math:`\\mathcal{M}_k` are exactly those writable as

.. math::

    Q(x) = \\sum_{|S| \\le k} h_S(x_S).

The m-projection of :math:`P` onto :math:`\\mathcal{M}_k` is

.. math::

    Q^{(k)} = \\arg\\min_{Q \\in \\mathcal{M}_k} D_{\\mathrm{KL}}(Q \\Vert P).

See Amari, "Information geometry on hierarchy of probability distributions"
(IEEE TIT, 2001).
"""

from copy import deepcopy
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from ..distribution import Distribution
from ..exceptions import ditException
from .optutil import prepare_dist

__all__ = (
    "m_projection",
    "m_projection_from_subsets",
    "m_projection_eps_limit",
    "mflat_mprojection_dists",
    "mflat_design_matrix",
    "mflat_design_matrix_from_subsets",
    "mflat_subsets_from_dependency",
    "symmetric_smooth",
)



def _outcome_tuples(dist):
    """Return outcomes as tuples of hashable symbols."""
    return [tuple(o) if not isinstance(o, tuple) else o for o in dist.outcomes]


def _alphabets(dist):
    """Per-variable alphabets from a dense Cartesian sample space."""
    n = dist.outcome_length()
    alphabets = []
    for i in range(n):
        alphabets.append(tuple(sorted({o[i] for o in _outcome_tuples(dist)})))
    return alphabets


def mflat_subsets_from_dependency(dependency, index_map=None):
    """
    Downward-closed collection of index-tuples for an antichain dependency.

    For a node :math:`\\pi`, the m-flat family is spanned by additive components
    on every nonempty :math:`S \\subseteq T` for some block :math:`T \\in \\pi`
    (plus a constant).

    Parameters
    ----------
    dependency : iterable of iterables
        Lattice node: an antichain of variable blocks (names or indices).
    index_map : dict or None
        Map from block element → integer coordinate index. Defaults to identity.

    Returns
    -------
    subsets : frozenset of tuple
        Sorted index tuples, all nonempty.
    """
    if index_map is None:
        index_map = {}
    subsets = set()
    for block in dependency:
        idxs = tuple(sorted(index_map.get(x, x) for x in block))
        for s in range(1, len(idxs) + 1):
            for sub in combinations(idxs, s):
                subsets.add(sub)
    return frozenset(subsets)


def mflat_design_matrix_from_subsets(alphabets, subsets):
    """
    Design matrix for the m-flat family spanned by the given variable subsets.

    Parameters
    ----------
    alphabets : sequence of sequences
        Per-variable alphabets.
    subsets : iterable of tuple
        Nonempty index tuples whose indicator features are free. The constant
        feature is always included.

    Returns
    -------
    A : np.ndarray
        Design matrix over the Cartesian product of ``alphabets``.
    outcomes : list of tuple
        Row-aligned outcomes.
    """
    n = len(alphabets)
    outcomes = list(product(*alphabets))
    n_out = len(outcomes)
    index = {o: i for i, o in enumerate(outcomes)}

    columns = [np.ones(n_out, dtype=float)]
    for subset in sorted(subsets, key=lambda s: (len(s), s)):
        subset = tuple(subset)
        if not subset or any(i < 0 or i >= n for i in subset):
            continue
        sub_alph = [alphabets[i] for i in subset]
        free = [i for i in range(n) if i not in subset]
        for assignment in product(*sub_alph):
            col = np.zeros(n_out, dtype=float)
            if not free:
                full = [None] * n
                for j, v in zip(subset, assignment, strict=True):
                    full[j] = v
                col[index[tuple(full)]] = 1.0
            else:
                free_alph = [alphabets[i] for i in free]
                for free_vals in product(*free_alph):
                    full = [None] * n
                    for j, v in zip(subset, assignment, strict=True):
                        full[j] = v
                    for j, v in zip(free, free_vals, strict=True):
                        full[j] = v
                    col[index[tuple(full)]] = 1.0
            columns.append(col)

    A = np.column_stack(columns) if columns else np.ones((n_out, 1))
    return A, outcomes


def mflat_design_matrix(alphabets, order):
    """
    Build the linear design matrix for the order-``order`` m-flat family.

    Columns are indicator features :math:`1_{x_S = a}` for every subset
    :math:`S` with :math:`|S| \\le` ``order`` and every assignment ``a`` on the
    corresponding alphabets. Any pmf in :math:`\\mathcal{M}_{\\mathrm{order}}`
    is then :math:`q = A\\theta` for some coefficient vector ``θ`` (subject to
    nonnegativity and normalization of ``q``).

    Parameters
    ----------
    alphabets : sequence of sequences
        The alphabet of each variable.
    order : int
        Maximum interaction order kept in the additive pmf expansion.

    Returns
    -------
    A : np.ndarray, shape (n_outcomes, n_features)
        Design matrix over the Cartesian product of ``alphabets``.
    outcomes : list of tuple
        Row-aligned Cartesian outcomes.
    """
    n = len(alphabets)
    if order < 0:
        msg = "order must be nonnegative"
        raise ditException(msg)
    if order > n:
        order = n

    subsets = []
    for s in range(1, order + 1):
        subsets.extend(combinations(range(n), s))
    return mflat_design_matrix_from_subsets(alphabets, subsets)


def _reverse_kl(q, p, eps=1e-16):
    """
    Compute :math:`D_{\\mathrm{KL}}(q \\Vert p)` in bits for dense linear pmfs.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    # xlogy handles 0 log 0; where p is tiny and q is not, contribution → +∞.
    if np.any((q > eps) & (p <= eps)):
        return np.inf
    return float(np.sum(xlogy(q, q) - xlogy(q, np.maximum(p, eps))) / np.log(2))


def _theta_for_pmf(A, q):
    """Least-squares coefficients realizing ``q`` approximately as ``A θ``."""
    theta, *_ = np.linalg.lstsq(A, q, rcond=None)
    return theta


def _forward_kl(p, q, eps=1e-300):
    """Forward KL :math:`D(P \\Vert Q)` in bits for dense linear pmfs."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], eps))))


def _jsd(p, q):
    """Jensen–Shannon divergence in bits for dense linear pmfs."""
    from ..divergences.jensen_shannon_divergence import jensen_shannon_divergence_pmf

    return float(jensen_shannon_divergence_pmf(np.vstack([p, q])))


def _divergence(p, q, criterion):
    """Dispatch divergence used as a projection objective / residual."""
    if criterion == "reverse_kl":
        val = _reverse_kl(q, p)
        return val if np.isfinite(val) else 1e6
    if criterion == "forward_kl":
        return _forward_kl(p, q)
    if criterion == "jsd":
        return _jsd(p, q)
    msg = f"unknown criterion {criterion!r}; expected reverse_kl, forward_kl, or jsd"
    raise ditException(msg)


def _project_onto_mflat(
    pmf,
    A,
    nrestarts=16,
    maxiter=3000,
    tol=1e-14,
    seed=0,
    warm_pmf=None,
    criterion="reverse_kl",
):
    """
    Project onto ``{q = A θ : q ≥ 0, ∑q = 1}`` under ``criterion``.

    Parameters
    ----------
    pmf : np.ndarray
        Target dense linear pmf (same row order as ``A``).
    A : np.ndarray
        Design matrix for :math:`\\mathcal{M}_k`.
    nrestarts, maxiter, tol, seed
        Optimizer controls.
    warm_pmf : np.ndarray or None
        Optional previous-ladder pmf used as an extra warm start.
    criterion : {'reverse_kl', 'forward_kl', 'jsd'}
        Projection objective. ``reverse_kl`` is the Amari m-projection;
        ``jsd`` is preferred for sparse supports (finite with no smoothing).

    Returns
    -------
    q : np.ndarray
        Optimized pmf on the same sample space.
    """
    n_out, n_feat = A.shape
    rng = np.random.default_rng(seed)

    def feasible_pmf(theta):
        q = A @ theta
        if np.any(q < -1e-10):
            q = np.maximum(q, 0.0)
        s = q.sum()
        if s <= 0:
            return None
        return q / s

    def objective(theta):
        q = feasible_pmf(theta)
        if q is None:
            return 1e6
        return _divergence(pmf, q, criterion)

    cons = [{"type": "eq", "fun": lambda th: float((A @ th).sum() - 1.0)}]
    for i in range(n_out):
        cons.append({"type": "ineq", "fun": lambda th, i=i: float((A @ th)[i])})

    initials = [_theta_for_pmf(A, pmf), np.zeros(n_feat)]
    theta_u = np.zeros(n_feat)
    theta_u[0] = 1.0 / n_out
    initials.append(theta_u)
    if warm_pmf is not None:
        initials.insert(0, _theta_for_pmf(A, warm_pmf))
    for _ in range(max(0, nrestarts - len(initials))):
        initials.append(rng.normal(0.0, 0.05, size=n_feat))

    best_q = None
    best_val = np.inf
    for theta0 in initials:
        q0 = A @ theta0
        if abs(q0.sum()) > 0:
            theta0 = theta0 / q0.sum()
        res = minimize(
            objective,
            theta0,
            method="SLSQP",
            constraints=cons,
            options={"maxiter": maxiter, "ftol": tol, "disp": False},
        )
        q = feasible_pmf(res.x)
        if q is None:
            continue
        val = _divergence(pmf, q, criterion)
        if val < best_val:
            best_val = val
            best_q = q

    if best_q is None:
        msg = "m-projection failed to find a feasible point in M_k"
        raise ditException(msg)
    return best_q



def symmetric_smooth(dist, eps):
    """
    Symmetric full-support smoothing :math:`P_\\varepsilon = (1-\\varepsilon)P + \\varepsilon U`.

    Here :math:`U` is uniform on the Cartesian product of the alphabets. This
    preserves permutation symmetries of ``dist`` (e.g. W).

    Parameters
    ----------
    dist : Distribution
    eps : float
        Mixture weight on the uniform. Must satisfy ``0 <= eps < 1``.
        ``eps=0`` returns a dense copy of ``dist`` (still sparse if ``dist`` is).

    Returns
    -------
    smoothed : Distribution
        Dense linear distribution on the Cartesian sample space.
    """
    eps = float(eps)
    if eps < 0 or eps >= 1:
        msg = "eps must satisfy 0 <= eps < 1"
        raise ditException(msg)

    d = prepare_dist(deepcopy(dist))
    alphabets = _alphabets(d)
    outcomes = list(product(*alphabets))
    p = _aligned_pmf(d, outcomes)
    if eps == 0:
        return _dist_from_pmf(outcomes, p, d)
    u = np.ones(len(outcomes), dtype=float) / len(outcomes)
    pe = (1.0 - eps) * p + eps * u
    return _dist_from_pmf(outcomes, pe, d)


def _resolve_reverse_kl_target(dist, eps=None):
    """
    Build the reverse-KL target: symmetric :math:`P_\\varepsilon`.

    If ``eps`` is ``None``, defaults to ``1e-8``.
    """
    if eps is None:
        eps = 1e-8
    return symmetric_smooth(dist, eps), float(eps)


def _aligned_pmf(dist, outcomes):
    """Dense linear pmf of ``dist`` aligned to ``outcomes`` order."""
    pmf_map = {tuple(o) if not isinstance(o, tuple) else o: p for o, p in zip(dist.outcomes, dist.pmf, strict=True)}
    pmf = np.array([pmf_map.get(o, 0.0) for o in outcomes], dtype=float)
    if pmf.sum() <= 0:
        msg = "target distribution has empty pmf"
        raise ditException(msg)
    return pmf / pmf.sum()


def _dist_from_pmf(outcomes, pmf, template):
    """Build a Distribution with optional RV names copied from ``template``."""
    q = Distribution(outcomes, pmf, base="linear", validate=False)
    q.normalize()
    if template.get_rv_names() is not None:
        q.set_rv_names(template.get_rv_names())
    return q


def m_projection(
    dist,
    order,
    nrestarts=16,
    maxiter=3000,
    warm_start=None,
    criterion="reverse_kl",
    eps=None,
):
    """
    Project ``dist`` onto the order-``order`` m-flat mixture family.

    Parameters
    ----------
    dist : Distribution
        The target distribution :math:`P`.
    order : int
        Maximum ANOVA order kept in :math:`\\mathcal{M}_{\\mathrm{order}}`.
    nrestarts, maxiter : int
        Optimizer controls.
    warm_start : Distribution or None
        Optional warm start.
    criterion : {'reverse_kl', 'forward_kl', 'jsd'}
        Projection objective.
    eps : float or None
        For ``reverse_kl``, weight in :math:`P_\\varepsilon=(1-\\varepsilon)P+\\varepsilon U`.
        Defaults to ``1e-8`` when ``None``.

    Returns
    -------
    q : Distribution
    """
    n = dist.outcome_length()
    if order < 0:
        msg = "order must be nonnegative"
        raise ditException(msg)
    if order > n:
        order = n
    subsets = [sub for s in range(1, order + 1) for sub in combinations(range(n), s)]
    return m_projection_from_subsets(
        dist,
        subsets,
        nrestarts=nrestarts,
        maxiter=maxiter,
        warm_start=warm_start,
        criterion=criterion,
        eps=eps,
    )


def m_projection_from_subsets(
    dist,
    subsets,
    nrestarts=16,
    maxiter=3000,
    warm_start=None,
    criterion="reverse_kl",
    eps=None,
):
    """
    Project ``dist`` onto the m-flat family spanned by ``subsets``.
    """
    if criterion == "reverse_kl":
        target, _eps = _resolve_reverse_kl_target(dist, eps=eps)
    else:
        target = prepare_dist(deepcopy(dist))

    n = target.outcome_length()
    alphabets = _alphabets(target)
    subsets = frozenset(tuple(s) for s in subsets if s)

    if frozenset(range(n)) in {frozenset(s) for s in subsets}:
        outcomes = list(product(*alphabets))
        return _dist_from_pmf(outcomes, _aligned_pmf(target, outcomes), target)

    A, outcomes = mflat_design_matrix_from_subsets(alphabets, subsets)
    if criterion == "reverse_kl":
        pmf = _aligned_pmf(target, outcomes)
    else:
        dense = prepare_dist(deepcopy(dist))
        pmf_map = {
            tuple(o) if not isinstance(o, tuple) else o: float(p)
            for o, p in zip(dense.outcomes, dense.pmf, strict=True)
        }
        pmf = np.array([pmf_map.get(o, 0.0) for o in outcomes], dtype=float)
        pmf = pmf / pmf.sum()

    if not subsets:
        q_pmf = np.ones(len(outcomes), dtype=float) / len(outcomes)
    else:
        warm_pmf = None if warm_start is None else _aligned_pmf(warm_start, outcomes)
        q_pmf = _project_onto_mflat(
            pmf,
            A,
            nrestarts=nrestarts,
            maxiter=maxiter,
            warm_pmf=warm_pmf,
            criterion=criterion,
        )

    return _dist_from_pmf(outcomes, q_pmf, target)


def m_projection_eps_limit(
    dist,
    subsets=None,
    order=None,
    eps_schedule=(1e-4, 1e-6, 1e-8),
    nrestarts=16,
    maxiter=3000,
    warm_start=None,
):
    """
    Reverse-KL m-projection with symmetric :math:`P_\\varepsilon`, taking
    :math:`\\varepsilon \\downarrow 0` along ``eps_schedule``.
    """
    from ..divergences import kullback_leibler_divergence as D

    schedule = tuple(float(e) for e in eps_schedule)
    if not schedule:
        msg = "eps_schedule must be nonempty"
        raise ditException(msg)

    if subsets is None:
        if order is None:
            order = dist.outcome_length()
        n = dist.outcome_length()
        subsets = [sub for s in range(1, order + 1) for sub in combinations(range(n), s)]

    path = []
    q = warm_start
    target = None
    for eps in schedule:
        q = m_projection_from_subsets(
            dist,
            subsets,
            eps=eps,
            nrestarts=nrestarts,
            maxiter=maxiter,
            warm_start=q,
            criterion="reverse_kl",
        )
        target = symmetric_smooth(dist, eps)
        path.append((eps, q))

    return {
        "dist": q,
        "target": target,
        "eps": schedule[-1],
        "rKL": float(D(q, target)),
        "path": path,
    }


def mflat_mprojection_dists(
    dist,
    k_max=None,
    nrestarts=16,
    maxiter=3000,
    criterion="reverse_kl",
    eps=None,
    eps_schedule=None,
):
    """
    Return the m-flat projection ladder :math:`Q^{(0)},\\ldots,Q^{(k_{\\max})}`.
    """
    n = dist.outcome_length()
    if k_max is None:
        k_max = n
    if k_max < 0:
        msg = "k_max must be nonnegative"
        raise ditException(msg)
    if k_max > n:
        k_max = n

    if criterion == "reverse_kl":
        if eps_schedule is not None:
            target = symmetric_smooth(dist, float(eps_schedule[-1]))
        else:
            target, _ = _resolve_reverse_kl_target(dist, eps=eps)
    else:
        target = prepare_dist(deepcopy(dist))

    alphabets = _alphabets(target)
    outcomes = list(product(*alphabets))

    dense = prepare_dist(deepcopy(dist))
    pmf_map = {
        tuple(o) if not isinstance(o, tuple) else o: float(p)
        for o, p in zip(dense.outcomes, dense.pmf, strict=True)
    }
    raw_pmf = np.array([pmf_map.get(o, 0.0) for o in outcomes], dtype=float)
    raw_pmf = raw_pmf / raw_pmf.sum()

    dists = []
    for k in range(k_max + 1):
        if k == 0:
            q = _dist_from_pmf(outcomes, np.ones(len(outcomes)) / len(outcomes), target)
        elif k == n:
            if criterion == "reverse_kl":
                q = _dist_from_pmf(outcomes, _aligned_pmf(target, outcomes), target)
            else:
                q = _dist_from_pmf(outcomes, raw_pmf, dist)
        else:
            warm = dists[-1] if dists else None
            if criterion == "reverse_kl" and eps_schedule is not None:
                q = m_projection_eps_limit(
                    dist,
                    order=k,
                    eps_schedule=eps_schedule,
                    nrestarts=nrestarts,
                    maxiter=maxiter,
                    warm_start=warm,
                )["dist"]
            else:
                q = m_projection(
                    dist,
                    k,
                    eps=eps,
                    nrestarts=nrestarts,
                    maxiter=maxiter,
                    warm_start=warm,
                    criterion=criterion,
                )
        dists.append(q)
    return dists
