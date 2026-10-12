"""
The functional common information.

F[X_0 : ... : X_{n-1} | Z] is the least entropy H[W] of a function W of the
variables such that the X_i are mutually independent given (W, Z)
:cite:`james2017multivariate`.

A function W is a partition of the support, and the X_i are independent given
(W = w, Z = z) exactly when the outcomes in block w with Z = z carry a product
distribution: their support is a combinatorial rectangle A_0 x ... x A_{n-1}
and the pmf factorizes over it. Since H[W] = sum_w -p(w) log2 p(w) is a sum of
per-block costs, F is a minimum-weight exact cover of the support by such
blocks, which is solved exactly as an integer program.

Without conditioning, every block lies within one connected component of the
support (outcomes are adjacent when they agree on some X_i), so W refines the
Gacs-Korner common variable and each component is solved independently.
"""

from collections import defaultdict
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ...distconst import insert_rvf
from ...exceptions import ditException
from ...helpers import as_linear, normalize_rvs
from ...utils import flatten, unitful
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy

__all__ = (
    "functional_common_information",
    "functional_variable",
)

# Bound on the number of candidate blocks handed to the integer program.
_MAX_CANDIDATES = 200_000


def _support(dist, rvs, crvs):
    """
    The support of (X_0, ..., X_{n-1}, Z) with its linear probabilities.

    Returns
    -------
    points : list of (tuple, tuple)
        Each point is ``(x, z)`` with ``x`` a tuple of per-variable symbols.
    probs : np.ndarray
        The probability of each point.
    members : list of list
        The outcomes of `dist` mapped to each point.
    """
    dims = list(dist.dims)

    def positions(rv):
        return [dims.index(name) for name in dist._resolve_rv_names(list(rv))]

    groups = [positions(rv) for rv in rvs]
    z_pos = positions(crvs) if crvs else []

    mass = defaultdict(float)
    members = defaultdict(list)
    for outcome, p in dist.zipped():
        if p <= 0:
            continue
        o = outcome if isinstance(outcome, tuple) else (outcome,)
        key = (tuple(tuple(o[i] for i in g) for g in groups), tuple(o[i] for i in z_pos))
        mass[key] += p
        members[key].append(o)

    points = sorted(mass, key=repr)
    return points, np.array([mass[pt] for pt in points]), [members[pt] for pt in points]


def _is_product(xs, ps):
    """
    Whether points `xs` (tuples) with probabilities `ps` form a product
    distribution on a combinatorial rectangle.
    """
    alphabets = [sorted({x[i] for x in xs}, key=repr) for i in range(len(xs[0]))]
    if np.prod([len(a) for a in alphabets]) != len(xs):
        return False
    index = [{v: j for j, v in enumerate(a)} for a in alphabets]
    q = np.zeros([len(a) for a in alphabets])
    for x, p in zip(xs, ps, strict=True):
        q[tuple(index[i][v] for i, v in enumerate(x))] = p
    q /= q.sum()
    outer = np.ones_like(q)
    for i in range(q.ndim):
        others = tuple(j for j in range(q.ndim) if j != i)
        outer = outer * q.sum(axis=others, keepdims=True)
    return bool(np.allclose(q, outer, rtol=1e-9, atol=1e-12))


def _rectangles(idx, points, probs):
    """
    All subsets of the points `idx` (sharing one value of Z) that carry a
    product distribution on a rectangle.

    Both conditions are inherited by sub-rectangles, so the valid rectangles
    are grown one symbol at a time from single points.

    Returns
    -------
    blocks : set of frozenset
        Each block is a set of indices into `points`.
    """
    lookup = {points[i][0]: i for i in idx}
    n = len(points[idx[0]][0])
    values = [sorted({points[i][0][k] for i in idx}, key=repr) for k in range(n)]

    def block_of(rect):
        cells = list(product(*rect))
        if any(c not in lookup for c in cells):
            return None
        block = [lookup[c] for c in cells]
        if len(block) > 1 and not _is_product(cells, probs[block]):
            return None
        return frozenset(block)

    seen = set()
    blocks = set()
    frontier = [tuple(frozenset([v]) for v in points[i][0]) for i in idx]
    while frontier:
        rect = frontier.pop()
        if rect in seen:
            continue
        seen.add(rect)
        block = block_of(rect)
        if block is None:
            continue
        blocks.add(block)
        if len(blocks) > _MAX_CANDIDATES:
            msg = "Too many candidate blocks for the functional common information."
            raise ditException(msg)
        for k in range(n):
            for v in values[k]:
                if v not in rect[k]:
                    grown = rect[:k] + (rect[k] | {v},) + rect[k + 1 :]
                    if grown not in seen:
                        frontier.append(grown)
    return blocks


def _components(points):
    """
    Connected components of the points, adjacent when they share a symbol of
    some X_i.
    """
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, (x, _) in enumerate(points):
        for k, v in enumerate(x):
            j = first.setdefault((k, v), i)
            parent[find(i)] = find(j)

    comps = defaultdict(list)
    for i in range(len(points)):
        comps[find(i)].append(i)
    return list(comps.values())


def _candidates(idx, points, probs, conditional):
    """
    Candidate blocks (sets of point indices) covering the points `idx`.

    With conditioning, a block of W may combine one rectangle from each value
    of Z.
    """
    by_z = defaultdict(list)
    for i in idx:
        by_z[points[i][1]].append(i)
    per_z = [sorted(_rectangles(sub, points, probs), key=sorted) for sub in by_z.values()]
    if not conditional:
        return per_z[0]

    n_combos = np.prod([len(r) + 1 for r in per_z], dtype=float) - 1
    if n_combos > _MAX_CANDIDATES:
        msg = "Too many candidate blocks for the conditional functional common information."
        raise ditException(msg)
    blocks = []
    for choice in product(*[[frozenset()] + r for r in per_z]):
        block = frozenset().union(*choice)
        if block:
            blocks.append(block)
    return blocks


def _min_entropy_cover(idx, blocks, probs):
    """
    The exact cover of `idx` by `blocks` minimizing sum_b -p(b) log2 p(b).

    Returns
    -------
    cost : float
        The entropy contribution of the chosen blocks.
    chosen : list of frozenset
        The chosen blocks.
    """
    blocks = list(blocks)
    masses = np.array([probs[list(b)].sum() for b in blocks])
    costs = -masses * np.log2(masses)
    if len(idx) == 1:
        return float(costs[0]), blocks

    row = {i: r for r, i in enumerate(idx)}
    A = np.zeros((len(idx), len(blocks)))
    for c, block in enumerate(blocks):
        for i in block:
            A[row[i], c] = 1
    res = milp(
        costs,
        constraints=LinearConstraint(A, 1, 1),
        integrality=np.ones(len(blocks)),
        bounds=Bounds(0, 1),
    )
    if not res.success:  # pragma: no cover
        msg = f"Exact cover failed: {res.message}"
        raise ditException(msg)
    chosen = [blocks[c] for c in np.flatnonzero(res.x > 0.5)]
    return float(costs[res.x > 0.5].sum()), chosen


def functional_markov_chain(dist, rvs=None, crvs=None, *, _stats=None):
    """
    Return H(W) for the least-entropy function W of `rvs` (and `crvs`) which
    renders `rvs` independent given `crvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the smallest function will be constructed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables to render independent. If None, then all random variables
        are used, which is equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    h : float
        The entropy of the least-entropy functional Markov variable W.
    """
    dist = as_linear(dist)
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    crvs = list(flatten(crvs))
    points, probs, members = _support(dist, rvs, crvs)

    conditional = bool(crvs)
    regions = [list(range(len(points)))] if conditional else _components(points)

    h = 0.0
    chosen = []
    for idx in regions:
        whole = frozenset(idx)
        # Merging blocks never increases entropy, so a valid whole region is optimal.
        if not conditional and (len(idx) == 1 or _is_product([points[i][0] for i in idx], probs[idx])):
            mass = probs[idx].sum()
            h += float(-mass * np.log2(mass))
            chosen.append(whole)
            continue
        cost, blocks = _min_entropy_cover(idx, _candidates(idx, points, probs, conditional), probs)
        h += cost
        chosen.extend(blocks)

    if _stats is not None:
        _stats["partition"] = frozenset(frozenset(outcome for i in block for outcome in members[i]) for block in chosen)
        _stats["components"] = len(regions)

    return h


@unitful
def functional_common_information(dist, rvs=None, crvs=None, return_partition=False):
    """
    Compute the functional common information, F, of `dist`. It is the entropy
    of the smallest random variable W such that all the variables in `rvs` are
    rendered independent conditioned on W (and `crvs`), and W is a function of
    `rvs` and `crvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the functional common information is
        computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    return_partition : bool
        If True, also return the optimal W as a partition of the outcomes of
        `dist`: a frozenset of blocks, each a frozenset of outcomes.

    Returns
    -------
    F : float
        The functional common information.
    partition : frozenset of frozensets
        The optimal W, only if `return_partition` is True.
    """
    dist = as_linear(dist)
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    # B = H means the identity on the support is optimal (unconditionally).
    if not crvs and not return_partition:
        dtc = dual_total_correlation(dist, rvs, crvs)
        if np.isclose(dtc, entropy(dist, rvs, crvs)):
            return dtc

    stats = {}
    h = functional_markov_chain(dist, rvs, crvs, _stats=stats)
    return (h, stats["partition"]) if return_partition else h


def functional_variable(dist, rvs=None, crvs=None):
    """
    Return `dist` with the optimal functional common variable W appended.

    W is the least-entropy function of `rvs` and `crvs` rendering `rvs`
    independent given `crvs`, so ``entropy(d, [d.outcome_length() - 1])`` is
    the functional common information. Its values are block labels ``0, 1,
    ...`` (as strings when the outcomes of `dist` have string symbols).

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables to render independent. If None, all random variables are
        used.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    d : Distribution
        The support of `dist` with W appended as its last variable.
    """
    dist = as_linear(dist).copy()
    dist.make_sparse()
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    stats = {}
    functional_markov_chain(dist, rvs, crvs, _stats=stats)

    blocks = sorted(stats["partition"], key=lambda block: min(map(repr, block)))
    as_str = all(isinstance(v, str) for o in dist.outcomes for v in (o if isinstance(o, tuple) else (o,)))
    label = {}
    for i, block in enumerate(blocks):
        for outcome in block:
            label[outcome] = str(i) if as_str else i

    def w(outcome):
        return (label[outcome if isinstance(outcome, tuple) else (outcome,)],)

    return insert_rvf(dist, w)
