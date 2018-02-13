"""
The variational distance.
"""

from __future__ import division

import numpy as np
from scipy.optimize import minimize_scalar

from ..exceptions import OptimizationException


def _normalize_pmfs(dist1, dist2):
    """
    Construct probability vectors with common support.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    p : np.ndarray
        The pmf of `dist1`.
    q : np.ndarray
        The pmf of `dist2`.
    """
    event_space = list(set().union(dist1.outcomes, dist2.outcomes))
    p = np.array([dist1[e] if e in dist1.outcomes else 0 for e in event_space])
    q = np.array([dist2[e] if e in dist2.outcomes else 0 for e in event_space])
    return p, q


def variational_distance_pmf(p, q):
    """
    Compute the variational distance.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    vd : float
        The variational distance.

    Notes
    -----
    `p` and `q` are assumed to be 1-to-1 with regards to events.
    """
    return abs(p-q).sum()/2


def variational_distance(dist1, dist2):
    """
    Compute the variational distance.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    vd : float
        The variational distance.
    """
    p, q = _normalize_pmfs(dist1, dist2)
    vd = variational_distance_pmf(p, q)
    return vd


def bhattacharyya_coefficient_pmf(p, q):
    """
    Compute the Bhattacharyya coefficient.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    bc : float
        The Bhattacharyya coefficient.
    """
    return np.sqrt(p*q).sum()


def bhattacharyya_coefficient(dist1, dist2):
    """
    Compute the Bhattacharyya coefficient.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    bc : float
        The Bhattacharyya coefficient.
    """
    p, q = _normalize_pmfs(dist1, dist2)
    bc = bhattacharyya_coefficient_pmf(p, q)
    return bc


def hellinger_distance_pmf(p, q):
    """
    Compute the Hellinger distance.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    hd : float
        The Hellinger distance.
    """
    bc = bhattacharyya_coefficient_pmf(p, q)
    hd = np.sqrt(1 - bc)
    hd = 0 if np.isnan(hd) else hd
    return hd


def hellinger_distance(dist1, dist2):
    """
    Compute the Hellinger distance.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    hd : float
        The Hellinger distance.
    """
    p, q = _normalize_pmfs(dist1, dist2)
    hd = hellinger_distance_pmf(p, q)
    return hd


def chernoff_information_pmf(p, q):
    """
    Compute the Chernoff information.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    ci : float
        The Chernoff information.
    """
    def func(alpha):
        return np.log2((p**alpha * q**(1-alpha)).sum())

    res = minimize_scalar(fun=func, bounds=(0, 1), method='bounded')

    if not -1e-8 <= res.x <= 1 + 1e-8:  # pragma: no cover
        msg = "Appropriate optima could not be found."
        raise OptimizationException(msg)

    ci = -func(res.x)
    # sometimes things are very slightly negative due to optimization fuzziness.
    # since this can throw off some inequalities, we set to zero in this case.
    if ci < 0:  # pragma: no cover
        ci = 0

    return ci


def chernoff_information(dist1, dist2):
    """
    Compute the Chernoff information.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    ci : float
        The Chernoff information.
    """
    p, q = _normalize_pmfs(dist1, dist2)
    ci = chernoff_information_pmf(p, q)
    return ci
