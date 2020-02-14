"""
Implementation of the Earth Mover's Distance.
"""

import numpy as np
from scipy.optimize import linprog

from ..helpers import normalize_pmfs, numerical_test


__all__ = (
    'categorical_distances',
    'earth_movers_distance',
    'earth_movers_distance_pmf',
    'numerical_distances',
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
    n = len(x)

    if distances is None:
        # assume categorical distribution
        distances = categorical_distances(n)

    eye = np.eye(n)
    A = np.vstack([np.dstack([eye] * n).reshape(n, n**2), np.tile(eye, n)])

    b = np.concatenate([x, y], axis=0)

    c = distances.flatten()

    res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])

    return res.fun


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
    if distances is None:
        try:
            numerical_test(dist1)
            numerical_test(dist2)
            p, q = dist1.pmf, dist2.pmf
            distances = numerical_distances(dist1.outcomes, dist2.outcomes)
        except TypeError:
            p, q = normalize_pmfs(dist1, dist2)
            distances = categorical_distances(len(p))
    else:
        p, q = dist1.pmf, dist2.pmf

    return earth_movers_distance_pmf(p, q, distances)
