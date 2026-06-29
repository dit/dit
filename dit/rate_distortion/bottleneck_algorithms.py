"""
Finite-alphabet algorithms for information bottleneck problems.
"""

from dataclasses import dataclass

import numpy as np

from ..exceptions import ditException
from .blahut_arimoto import blahut_arimoto_ib

__all__ = (
    "BottleneckResult",
    "active_alphabet",
    "agglomerative_ib",
    "bottleneck_result",
    "blahut_arimoto_ib",
    "sequential_ib",
)


@dataclass(frozen=True)
class BottleneckResult:
    """
    The result of a finite-alphabet information bottleneck optimization.

    Attributes
    ----------
    p_t_given_x : np.ndarray
        The encoder :math:`p(t | x)`.
    p_xyt : np.ndarray
        The induced joint distribution :math:`p(x, y, t)`.
    objective : float
        The Lagrangian objective value.
    complexity : float
        :math:`I(X : T)`.
    relevance : float
        :math:`I(T : Y)`.
    entropy : float
        :math:`H(T)`.
    distortion : float
        :math:`I(X : Y) - I(T : Y)`.
    iterations : int
        The number of update iterations.
    converged : bool
        Whether the optimizer reached a fixed point.
    active : int
        The number of non-empty bottleneck states.
    assignments : np.ndarray
        The hard cluster assignment for each :math:`x`, when available.
    history : tuple[float, ...]
        Objective values observed during optimization.
    active_history : tuple[int, ...]
        Active alphabet sizes observed during optimization.
    merges : tuple[tuple[int, int], ...]
        Agglomerative merge sequence, when available.
    """

    p_t_given_x: np.ndarray
    p_xyt: np.ndarray
    objective: float
    complexity: float
    relevance: float
    entropy: float
    distortion: float
    iterations: int
    converged: bool
    active: int
    assignments: np.ndarray | None = None
    history: tuple[float, ...] = ()
    active_history: tuple[int, ...] = ()
    merges: tuple[tuple[int, int], ...] = ()


def _validate_p_xy(p_xy):
    """
    Validate and normalize a finite joint distribution.
    """
    p_xy = np.asarray(p_xy, dtype=float)

    if p_xy.ndim != 2:
        msg = "p_xy must be a two-dimensional joint distribution."
        raise ditException(msg)
    if not np.all(np.isfinite(p_xy)):
        msg = "p_xy must contain only finite values."
        raise ditException(msg)
    if np.any(p_xy < 0):
        msg = "p_xy must be non-negative."
        raise ditException(msg)

    total = p_xy.sum()
    if total <= 0:
        msg = "p_xy must have positive total mass."
        raise ditException(msg)

    return p_xy / total


def _entropy(pmf):
    """
    Compute entropy in bits.
    """
    pmf = np.asarray(pmf, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.nansum(pmf * np.log2(pmf)))


def _mutual_information(joint):
    """
    Compute mutual information in bits from a two-dimensional joint pmf.
    """
    joint = np.asarray(joint, dtype=float)
    p0 = joint.sum(axis=1, keepdims=True)
    p1 = joint.sum(axis=0, keepdims=True)
    denom = p0 * p1
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(joint * np.log2(joint / denom))
    return float(mi)


def _p_xyt(p_xy, p_t_given_x):
    """
    Construct :math:`p(x, y, t)` from :math:`p(x, y)` and :math:`p(t | x)`.
    """
    return p_xy[:, :, np.newaxis] * p_t_given_x[:, np.newaxis, :]


def active_alphabet(p_t_given_x, p_x=None, tol=1e-12):
    """
    Count bottleneck symbols with non-zero probability.
    """
    p_t_given_x = np.asarray(p_t_given_x, dtype=float)
    p_t = p_t_given_x.sum(axis=0) if p_x is None else np.asarray(p_x, dtype=float) @ p_t_given_x
    return int((p_t > tol).sum())


def bottleneck_result(
    p_xy,
    p_t_given_x,
    beta,
    alpha=1.0,
    iterations=0,
    converged=True,
    assignments=None,
    history=(),
    active_history=(),
    merges=(),
):
    """
    Evaluate a bottleneck encoder and return a :class:`BottleneckResult`.
    """
    p_xy = _validate_p_xy(p_xy)
    p_t_given_x = np.asarray(p_t_given_x, dtype=float)

    if p_t_given_x.ndim != 2 or p_t_given_x.shape[0] != p_xy.shape[0]:
        msg = "p_t_given_x must have shape (|X|, |T|)."
        raise ditException(msg)
    if np.any(p_t_given_x < 0) or not np.all(np.isfinite(p_t_given_x)):
        msg = "p_t_given_x must contain finite non-negative values."
        raise ditException(msg)

    rows = p_t_given_x.sum(axis=1, keepdims=True)
    if np.any(np.isclose(rows, 0.0)):
        msg = "each row of p_t_given_x must have positive mass."
        raise ditException(msg)
    p_t_given_x = p_t_given_x / rows

    p_xyt = _p_xyt(p_xy, p_t_given_x)
    p_xt = p_xyt.sum(axis=1)
    p_ty = p_xyt.sum(axis=0).T

    complexity = _mutual_information(p_xt)
    relevance = _mutual_information(p_ty)
    entropy = _entropy(p_xt.sum(axis=0))
    other = entropy - complexity
    distortion = _mutual_information(p_xy) - relevance
    objective = entropy - alpha * other + beta * distortion

    return BottleneckResult(
        p_t_given_x=p_t_given_x,
        p_xyt=p_xyt,
        objective=float(objective),
        complexity=float(complexity),
        relevance=float(relevance),
        entropy=float(entropy),
        distortion=float(distortion),
        iterations=int(iterations),
        converged=bool(converged),
        active=active_alphabet(p_t_given_x, p_xy.sum(axis=1)),
        assignments=None if assignments is None else np.asarray(assignments, dtype=int),
        history=tuple(float(v) for v in history),
        active_history=tuple(int(v) for v in active_history),
        merges=tuple(tuple(int(i) for i in merge) for merge in merges),
    )


def _compact_assignments(assignments, p_x, tol=1e-12):
    """
    Relabel active positive-mass clusters contiguously.
    """
    assignments = np.asarray(assignments, dtype=int)
    p_x = np.asarray(p_x, dtype=float)
    labels = []
    for label in np.unique(assignments):
        if p_x[assignments == label].sum() > tol:
            labels.append(label)

    if not labels:
        return np.zeros_like(assignments)

    remap = {label: i for i, label in enumerate(labels)}
    compacted = np.zeros_like(assignments)
    for i, label in enumerate(assignments):
        compacted[i] = remap.get(label, 0)
    return compacted


def _encoder_from_assignments(assignments, n_clusters=None):
    """
    Build a deterministic encoder from hard assignments.
    """
    assignments = np.asarray(assignments, dtype=int)
    if np.any(assignments < 0):
        msg = "assignments must be non-negative integers."
        raise ditException(msg)

    if n_clusters is None:
        n_clusters = int(assignments.max()) + 1 if len(assignments) else 0
    if n_clusters <= 0:
        msg = "n_clusters must be positive."
        raise ditException(msg)
    if np.any(assignments >= n_clusters):
        msg = "assignments contain a cluster outside n_clusters."
        raise ditException(msg)

    p_t_given_x = np.zeros((len(assignments), n_clusters))
    p_t_given_x[np.arange(len(assignments)), assignments] = 1.0
    return p_t_given_x


def _result_from_assignments(
    p_xy,
    assignments,
    beta,
    iterations=0,
    converged=True,
    history=(),
    active_history=(),
    merges=(),
):
    """
    Compact hard assignments and evaluate the deterministic bottleneck objective.
    """
    p_xy = _validate_p_xy(p_xy)
    compacted = _compact_assignments(assignments, p_xy.sum(axis=1))
    p_t_given_x = _encoder_from_assignments(compacted)
    return bottleneck_result(
        p_xy=p_xy,
        p_t_given_x=p_t_given_x,
        beta=beta,
        alpha=0.0,
        iterations=iterations,
        converged=converged,
        assignments=compacted,
        history=history,
        active_history=active_history,
        merges=merges,
    )


def _initial_assignments(n_x, n_clusters, restart, rng):
    """
    Construct deterministic and random initial hard cluster assignments.
    """
    if restart == 0:
        assignments = np.arange(n_x) % n_clusters
    elif restart == 1:
        assignments = np.zeros(n_x, dtype=int)
    else:
        assignments = rng.integers(n_clusters, size=n_x)
    return assignments.astype(int)


def sequential_ib(
    p_xy,
    beta,
    n_clusters=None,
    initial_assignments=None,
    max_iters=100,
    restarts=25,
    tol=1e-10,
    random_state=None,
):
    r"""
    Compute a deterministic information bottleneck by hard reassignment.

    This is a finite-alphabet coordinate descent solver for the deterministic
    information bottleneck Lagrangian

    .. math::

        H(T) + \beta \left[I(X : Y) - I(T : Y)\right].

    Parameters
    ----------
    p_xy : np.ndarray
        Joint distribution over :math:`X` and :math:`Y`.
    beta : float
        Non-negative bottleneck Lagrange multiplier.
    n_clusters : int, None
        Maximum number of bottleneck states. If None, use ``|X|``.
    initial_assignments : np.ndarray, None
        Optional hard assignments to use as the first restart.
    max_iters : int
        Maximum coordinate-descent sweeps per restart.
    restarts : int
        Number of random restarts.
    tol : float
        Minimum objective improvement required to accept a reassignment.
    random_state : int, np.random.Generator, None
        Random seed or generator for stochastic restarts.

    Returns
    -------
    result : BottleneckResult
        The best hard bottleneck found.
    """
    p_xy = _validate_p_xy(p_xy)
    if beta < 0:
        msg = "beta must be non-negative."
        raise ditException(msg)

    n_x = p_xy.shape[0]
    if n_clusters is None:
        n_clusters = n_x
    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= n_x:
        msg = "n_clusters must be between 1 and |X|."
        raise ditException(msg)
    if max_iters < 0:
        msg = "max_iters must be non-negative."
        raise ditException(msg)
    if restarts < 0:
        msg = "restarts must be non-negative."
        raise ditException(msg)

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    starts = []
    if initial_assignments is not None:
        initial_assignments = np.asarray(initial_assignments, dtype=int)
        if initial_assignments.shape != (n_x,):
            msg = "initial_assignments must have shape (|X|,)."
            raise ditException(msg)
        if np.any(initial_assignments < 0):
            msg = "initial_assignments must be non-negative."
            raise ditException(msg)
        starts.append(initial_assignments % n_clusters)

    for restart in range(max(0, restarts)):
        starts.append(_initial_assignments(n_x, n_clusters, restart, rng))
    if not starts:
        starts.append(_initial_assignments(n_x, n_clusters, 0, rng))

    best = None
    for start in starts:
        assignments = np.asarray(start, dtype=int).copy()
        result = _result_from_assignments(p_xy, assignments, beta)
        history = [result.objective]
        active_history = [result.active]
        converged = False
        iterations = 0

        for _iteration in range(1, max_iters + 1):
            iterations = _iteration
            changed = False
            for x in range(n_x):
                current_label = assignments[x]
                current = _result_from_assignments(p_xy, assignments, beta).objective
                best_label = current_label
                best_objective = current

                for label in range(n_clusters):
                    if label == current_label:
                        continue
                    candidate = assignments.copy()
                    candidate[x] = label
                    objective = _result_from_assignments(p_xy, candidate, beta).objective
                    if objective < best_objective - tol:
                        best_objective = objective
                        best_label = label

                if best_label != current_label:
                    assignments[x] = best_label
                    changed = True

            result = _result_from_assignments(p_xy, assignments, beta)
            history.append(result.objective)
            active_history.append(result.active)

            if not changed:
                converged = True
                break

        result = _result_from_assignments(
            p_xy=p_xy,
            assignments=assignments,
            beta=beta,
            iterations=iterations,
            converged=converged,
            history=history,
            active_history=active_history,
        )
        if best is None or result.objective < best.objective:
            best = result

    return best


def agglomerative_ib(p_xy, beta=1.0, n_clusters=None):
    """
    Greedily merge hard clusters for the deterministic information bottleneck.

    The algorithm starts from the singleton partition of :math:`X` and repeatedly
    merges the pair of clusters that gives the lowest next DIB objective. If
    ``n_clusters`` is None, the returned result is the best point along the full
    greedy merge path.

    Parameters
    ----------
    p_xy : np.ndarray
        Joint distribution over :math:`X` and :math:`Y`.
    beta : float
        Non-negative bottleneck Lagrange multiplier.
    n_clusters : int, None
        Target number of clusters. If None, return the best partition on the
        full dendrogram.

    Returns
    -------
    result : BottleneckResult
        The selected hard bottleneck result. Its ``history``,
        ``active_history``, and ``merges`` fields describe the greedy path.
    """
    p_xy = _validate_p_xy(p_xy)
    if beta < 0:
        msg = "beta must be non-negative."
        raise ditException(msg)

    n_x = p_xy.shape[0]
    if n_clusters is not None:
        n_clusters = int(n_clusters)
        if not 1 <= n_clusters <= n_x:
            msg = "n_clusters must be between 1 and |X|."
            raise ditException(msg)

    assignments = np.arange(n_x, dtype=int)
    results = [_result_from_assignments(p_xy, assignments, beta)]
    merges = []

    target = 1 if n_clusters is None else n_clusters
    while results[-1].active > target:
        labels = np.unique(assignments)
        best_assignments = None
        best_pair = None
        best_result = None

        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                candidate = assignments.copy()
                candidate[candidate == right] = left
                candidate = _compact_assignments(candidate, p_xy.sum(axis=1))
                result = _result_from_assignments(p_xy, candidate, beta)
                if best_result is None or result.objective < best_result.objective:
                    best_result = result
                    best_assignments = candidate
                    best_pair = (left, right)

        assignments = best_assignments
        merges.append(best_pair)
        results.append(best_result)

    if n_clusters is None:
        selected_index, selected = min(enumerate(results), key=lambda item: item[1].objective)
    else:
        selected_index, selected = len(results) - 1, results[-1]

    history = [result.objective for result in results]
    active_history = [result.active for result in results]
    selected_merges = merges[:selected_index]

    return _result_from_assignments(
        p_xy=p_xy,
        assignments=selected.assignments,
        beta=beta,
        iterations=selected_index,
        converged=True,
        history=history,
        active_history=active_history,
        merges=selected_merges,
    )
