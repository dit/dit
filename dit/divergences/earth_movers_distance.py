"""
Implementation of the Earth Mover's Distance.
"""

import numpy as np
from scipy.optimize import linprog

from ..distribution import Distribution
from ..helpers import numerical_test

__all__ = (
    "categorical_distances",
    "earth_movers_distance",
    "earth_movers_distance_coupling",
    "earth_movers_distance_pmf",
    "numerical_distances",
)


def categorical_distances(n):
    """
    Construct a categorical distances matrix.

    Parameters
    ----------
    n : int
        The size of the matrix.

    Returns
    -------
    ds : np.ndarray
        The matrix of distances.
    """
    return 1 - np.eye(n)


def numerical_distances(x_values, y_values):
    """
    Construct matrix of distances between real values.

    Parameters
    ----------
    x_values : np.ndarray
        The real values on the x dimension.
    y_values : np.ndarray
        The real values on the y dimension.

    Returns
    -------
    ds : np.ndarray
        The matrix of distances.
    """
    xx, yy = np.meshgrid(x_values, y_values)
    return abs(xx - yy)


def _emd_linprog(x, y, distances=None):
    """
    Solve the optimal-transport LP between `x` and `y`.

    Parameters
    ----------
    x : np.ndarray
        The first pmf.
    y : np.ndarray
        The second pmf.
    distances : np.ndarray, None
        The cost of moving probability from x[i] to y[j]. If None,
        the cost is assumed to be i != j.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        The result of the linear program. ``res.fun`` is the Earth Mover's
        Distance; ``res.x.reshape(len(x), len(y))`` is the optimal transport
        plan (a joint distribution with `x` as row marginal and `y` as column
        marginal).
    """
    n = len(x)

    if distances is None:
        # assume categorical distribution
        distances = categorical_distances(n)

    eye = np.eye(n)
    A = np.vstack([np.dstack([eye] * n).reshape(n, n**2), np.tile(eye, n)])

    b = np.concatenate([x, y], axis=0)

    c = distances.flatten()

    return linprog(c, A_eq=A, b_eq=b, bounds=[0, None])


def earth_movers_distance_pmf(x, y, distances=None):
    """
    Compute the Earth Mover's Distance between `p` and `q`.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.
    distances : np.ndarray, None
        The cost of moving probability from p[i] to q[j]. If None,
        the cost is assumed to be i != j.

    Returns
    -------
    emd : float
        The Earth Mover's Distance.
    """
    return _emd_linprog(x, y, distances).fun


def _emd_setup(dist1, dist2, distances=None):
    """
    Resolve pmfs, a distance matrix, and row/column outcome labels.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution (row marginal).
    dist2 : Distribution
        The second distribution (column marginal).
    distances : np.ndarray, None
        A matrix of distances between outcomes. If None, one is constructed
        as in :func:`earth_movers_distance`.

    Returns
    -------
    p, q : np.ndarray
        The row and column pmfs.
    distances : np.ndarray
        The distance matrix.
    outcomes1, outcomes2 : tuple
        The outcome labels indexing the rows (`p`) and columns (`q`).
    """
    if distances is None:
        try:
            numerical_test(dist1)
            numerical_test(dist2)
            p, q = dist1.pmf, dist2.pmf
            distances = numerical_distances(dist1.outcomes, dist2.outcomes)
            outcomes1, outcomes2 = dist1.outcomes, dist2.outcomes
        except TypeError:
            event_space = list(set().union(dist1.outcomes, dist2.outcomes))
            p = np.array([dist1[e] if e in dist1.outcomes else 0 for e in event_space])
            q = np.array([dist2[e] if e in dist2.outcomes else 0 for e in event_space])
            distances = categorical_distances(len(p))
            outcomes1 = outcomes2 = tuple(event_space)
    else:
        p, q = dist1.pmf, dist2.pmf
        outcomes1, outcomes2 = dist1.outcomes, dist2.outcomes

    return p, q, distances, outcomes1, outcomes2


def earth_movers_distance(dist1, dist2, distances=None):
    """
    Compute the Earth Mover's Distance (EMD) between `dist1` and `dist2`. The
    EMD is the least amount of "probability mass flow" that must occur to
    transform `dist1` to `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.
    distances : np.ndarray, None
        A matrix of distances between outcomes of the distributions.
        If None, a distance matrix is constructed; if the distributions
        are categorical each non-equal event is considered at unit distance,
        and if numerical abs(x, y) is used as the distance.

    Returns
    -------
    emd : float
        The Earth Mover's Distance.
    """
    p, q, distances, _, _ = _emd_setup(dist1, dist2, distances)

    return earth_movers_distance_pmf(p, q, distances)


def earth_movers_distance_coupling(dist1, dist2, distances=None):
    """
    Compute the optimal transport coupling underlying the Earth Mover's
    Distance between `dist1` and `dist2`.

    The Earth Mover's Distance is the minimal expected ground-metric cost over
    all joint distributions (couplings) whose marginals are `dist1` and
    `dist2`. This returns the minimizing coupling itself: a joint distribution
    with `dist1` as its first marginal and `dist2` as its second.

    .. note::

        The optimal coupling is not unique in general; the linear program
        returns a single optimal vertex.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution (first marginal of the coupling).
    dist2 : Distribution
        The second distribution (second marginal of the coupling).
    distances : np.ndarray, None
        A matrix of distances between outcomes of the distributions.
        If None, a distance matrix is constructed; if the distributions
        are categorical each non-equal event is considered at unit distance,
        and if numerical abs(x, y) is used as the distance.

    Returns
    -------
    coupling : Distribution
        The optimal transport plan as a joint distribution over pairs of
        outcomes ``(o1, o2)``, with ``dist1`` and ``dist2`` as its marginals.
    """
    p, q, distances, outcomes1, outcomes2 = _emd_setup(dist1, dist2, distances)

    plan = _emd_linprog(p, q, distances).x.reshape(len(p), len(q))

    outcomes = [_as_tuple(o1) + _as_tuple(o2) for o1 in outcomes1 for o2 in outcomes2]
    return Distribution(outcomes, plan.reshape(-1), trim=False)


def _as_tuple(outcome):
    """
    Coerce a distribution outcome into a tuple of random-variable values.
    """
    return outcome if isinstance(outcome, tuple) else (outcome,)
