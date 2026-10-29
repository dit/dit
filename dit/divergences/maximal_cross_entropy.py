r"""
The maximal cross entropy over couplings.

For a target joint distribution :math:`q(x, y)` and a pair of marginals
:math:`p_X, p_Y`, the maximal cross entropy :cite:`yu2020exact` is the largest
cross entropy, relative to :math:`q`, of any joint distribution with those
marginals:

.. math::

   \mathbb{H}(p_X, p_Y \| q) = \max_{r \in \mathcal{C}(p_X, p_Y)} -\sum_{x, y} r(x, y) \log_2 q(x, y),

where :math:`\mathcal{C}(p_X, p_Y)` is the set of couplings of :math:`p_X` and
:math:`p_Y`. It is a transportation problem, so it is solved as a linear
program.
"""

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..utils import unitful

__all__ = (
    "maximal_cross_entropy",
    "maximal_cross_entropy_pmf",
)


@lru_cache(maxsize=64)
def _coupling_constraints(m, n, k):
    """
    The equality constraints fixing the marginals of `k` stacked `m` by `n`
    couplings: for each coupling, its `m` row sums and then its first `n - 1`
    column sums. The last column sum is implied by the others, and leaving it out
    keeps the solver from declaring the problem infeasible when the two marginals'
    totals differ by rounding.
    """
    block = sp.vstack([sp.kron(sp.eye(m), np.ones((1, n))), sp.kron(np.ones((1, m)), sp.eye(n), format="csr")[: n - 1]])
    return sp.block_diag([block] * k, format="csr")


def _transport(a, b, costs, maximize, forbidden=None):
    """
    Solve the transportation problems of each pair of marginals ``(a[w], b[w])``
    as one linear program.

    Parameters
    ----------
    a : np.ndarray, shape (k, m)
        Row marginals, one per coupling.
    b : np.ndarray, shape (k, n)
        Column marginals; ``b[w]`` must have the same total mass as ``a[w]``.
    costs : np.ndarray, shape (k, m, n)
        Finite costs.
    maximize : bool
        Whether to maximize, rather than minimize, the expected cost.
    forbidden : np.ndarray, None
        Boolean array broadcastable to (k, m, n) of cells that may carry no
        mass. The problem must remain feasible.

    Returns
    -------
    plans : np.ndarray, shape (k, m, n)
        The optimal couplings.
    f : np.ndarray, shape (k, m)
        Optimal dual potentials of the row marginals.
    g : np.ndarray, shape (k, n)
        Optimal dual potentials of the column marginals, normalized so that
        ``g[:, -1] == 0``. The optimal value of coupling ``w`` is
        ``a[w] @ f[w] + b[w] @ g[w]``, and ``f[w][x] + g[w][y]`` is its
        (super- or sub-) gradient with respect to mass added at ``(x, y)``.
    """
    k, m = a.shape
    n = b.shape[1]
    sign = -1.0 if maximize else 1.0
    bounds = (0, None)
    if forbidden is not None:
        upper = np.where(np.broadcast_to(forbidden, (k, m, n)), 0.0, np.inf).ravel()
        bounds = np.column_stack([np.zeros_like(upper), upper])
    res = linprog(
        sign * np.asarray(costs).ravel(),
        A_eq=_coupling_constraints(m, n, k),
        b_eq=np.concatenate([a, b[:, : n - 1]], axis=1).ravel(),
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:  # pragma: no cover
        # Presolve occasionally misjudges problems with tiny marginals as infeasible.
        res = linprog(
            sign * np.asarray(costs).ravel(),
            A_eq=_coupling_constraints(m, n, k),
            b_eq=np.concatenate([a, b[:, : n - 1]], axis=1).ravel(),
            bounds=bounds,
            method="highs",
            options={"presolve": False},
        )
    if res.status != 0:  # pragma: no cover
        msg = f"The coupling linear program failed: {res.message}"
        raise ditException(msg)
    duals = sign * res.eqlin.marginals.reshape(k, m + n - 1)
    g = np.concatenate([duals[:, m:], np.zeros((k, 1))], axis=1)
    return res.x.reshape(k, m, n), duals[:, :m], g


def _max_coupling_binary(a, b, cost):
    """
    :func:`_max_coupling` for two rows, a fractional knapsack: column ``y``
    gains ``cost[0, y] - cost[1, y]`` per unit routed to the first row, so the
    first row's mass fills the columns in decreasing order of that gain. The
    gain of the column where the filling stops prices the first row's mass.
    """
    gain = cost[0] - cost[1]
    order = np.argsort(-gain, kind="stable")
    sorted_gain = gain[order]
    sorted_b = b[:, order]
    filled = np.cumsum(sorted_b, axis=1)
    routed = np.clip(a[:, :1] - (filled - sorted_b), 0.0, sorted_b)
    values = b @ cost[1] + routed @ sorted_gain
    reached = filled >= a[:, :1] - 1e-15
    stop = np.where(reached.any(axis=1), reached.argmax(axis=1), b.shape[1] - 1)
    price = sorted_gain[stop]
    f = np.column_stack([price, np.zeros_like(price)])
    g = cost[1] + np.maximum(gain - price[:, np.newaxis], 0.0)
    return values, f, g


def _max_coupling(a, b, cost):
    """
    Maximize the expected `cost` over the couplings of each pair of marginals
    ``(a[w], b[w])``. Pairs with a binary side are solved in closed form; the
    rest are solved as one linear program.

    Parameters
    ----------
    a : np.ndarray, shape (k, m)
        Row marginals, one per coupling.
    b : np.ndarray, shape (k, n)
        Column marginals; ``b[w]`` must have the same total mass as ``a[w]``.
    cost : np.ndarray, shape (m, n)
        Finite costs, shared by every coupling.

    Returns
    -------
    values : np.ndarray, shape (k,)
        The maximal expected cost of each coupling.
    f : np.ndarray, shape (k, m)
        Optimal dual potentials of the row marginals.
    g : np.ndarray, shape (k, n)
        Optimal dual potentials of the column marginals. ``f[w][x] + g[w][y]``
        is a supergradient of the (concave) value of coupling ``w`` with respect
        to mass added at ``(x, y)``.
    """
    if cost.shape[0] == 2:
        return _max_coupling_binary(a, b, cost)
    if cost.shape[1] == 2:
        values, g, f = _max_coupling_binary(b, a, cost.T)
        return values, f, g
    costs = np.broadcast_to(cost, (a.shape[0], *cost.shape))
    plans, f, g = _transport(a, b, costs, maximize=True)
    return (plans * costs).sum(axis=(1, 2)), f, g


def maximal_cross_entropy_pmf(px, py, pxy):
    r"""
    The maximal cross entropy of the couplings of `px` and `py` relative to
    `pxy`, in bits.

    Parameters
    ----------
    px : np.ndarray
        The first marginal; normalized before use.
    py : np.ndarray
        The second marginal; normalized before use.
    pxy : np.ndarray
        The target joint distribution, of shape ``(len(px), len(py))``.

    Returns
    -------
    xh : float
        :math:`\max_{r \in \mathcal{C}(p_X, p_Y)} -\sum r \log_2 q`, which is
        infinite when some coupling can place mass where `pxy` vanishes.
    """
    px, py, pxy = (np.asarray(v, dtype=float) for v in (px, py, pxy))
    rows, cols = px > 0, py > 0
    target = pxy[np.ix_(rows, cols)]
    if (target <= 0).any():
        return np.inf
    a, b = px[rows] / px.sum(), py[cols] / py.sum()
    values, _, _ = _max_coupling(a[np.newaxis], b[np.newaxis], -np.log2(target))
    return float(values[0])


def _coalesced(dist, rvs):
    """The probabilities of `dist` keyed by the pair of values its two groups `rvs` take."""
    d = dist.coalesce(rvs)
    d.make_sparse()
    probs = {}
    for o, p in zip(d.outcomes, d.pmf, strict=True):
        probs[o[0], o[1]] = probs.get((o[0], o[1]), 0.0) + p
    return probs


@unitful
def maximal_cross_entropy(dist1, dist2=None, rvs=None):
    r"""
    The maximal cross entropy :cite:`yu2020exact`: the largest cross entropy,
    relative to `dist2`, of any joint distribution whose marginals on the two
    groups of variables in `rvs` match those of `dist1`,

    .. math::

       \mathbb{H}(p_X, p_Y \| q) = \max_{r \in \mathcal{C}(p_X, p_Y)} -\sum_{x, y} r(x, y) \log_2 q(x, y).

    It is at least the cross entropy of any particular coupling, such as
    :math:`p_X p_Y`, and it is infinite when some coupling places mass where
    :math:`q` vanishes, that is, when the product of the supports of
    :math:`p_X` and :math:`p_Y` is not contained in the support of :math:`q`.
    With `dist2` equal to `dist1` it is at least the joint entropy, with equality
    exactly when the two groups are independent.

    Parameters
    ----------
    dist1 : Distribution
        The distribution whose marginals :math:`p_X` and :math:`p_Y` are coupled.
    dist2 : Distribution, None
        The target distribution :math:`q`. If None, `dist1` is used.
    rvs : list, None
        Two lists of indexes, the variables of :math:`X` and of :math:`Y`. If
        None, the distributions must have exactly two variables.

    Returns
    -------
    xh : float
        The maximal cross entropy.

    Raises
    ------
    ditException
        Raised if `rvs` does not specify exactly two groups of variables, or if a
        distribution is symbolic.
    """
    if dist2 is None:
        dist2 = dist1
    if dist1.is_symbolic() or dist2.is_symbolic():
        msg = "The maximal cross entropy is a linear program and does not support symbolic distributions."
        raise ditException(msg)
    if rvs is None:
        if dist1.outcome_length() != 2:
            msg = "The maximal cross entropy needs `rvs` unless the distribution has exactly two variables."
            raise ditException(msg)
        rvs = [[0], [1]]
    if len(rvs) != 2:
        msg = f"The maximal cross entropy couples two groups of variables, not {len(rvs)}."
        raise ditException(msg)
    rvs, _ = normalize_rvs(dist1, rvs, None)
    normalize_rvs(dist2, rvs, None)

    joint1 = _coalesced(dist1, rvs)
    joint2 = _coalesced(dist2, rvs)
    xs = sorted({x for x, _ in joint1}, key=repr)
    ys = sorted({y for _, y in joint1}, key=repr)
    px = np.array([sum(p for (x, _), p in joint1.items() if x == x0) for x0 in xs])
    py = np.array([sum(p for (_, y), p in joint1.items() if y == y0) for y0 in ys])
    pxy = np.array([[joint2.get((x, y), 0.0) for y in ys] for x in xs])

    return maximal_cross_entropy_pmf(px, py, pxy)
