"""
The variational distance.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from ..exceptions import OptimizationException
from ..helpers import normalize_pmfs
from ..utils import unitful

__all__ = (
    "bhattacharyya_angle",
    "bhattacharyya_angle_pmf",
    "bhattacharyya_coefficient",
    "bhattacharyya_coefficient_pmf",
    "bhattacharyya_distance",
    "bhattacharyya_distance_pmf",
    "chernoff_information",
    "chernoff_information_pmf",
    "fisher_rao_distance",
    "fisher_rao_distance_pmf",
    "hellinger_distance",
    "hellinger_distance_pmf",
    "matusita_distance",
    "matusita_distance_pmf",
    "variational_distance",
    "variational_distance_pmf",
)


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
    return abs(p - q).sum() / 2


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
    p, q = normalize_pmfs(dist1, dist2)
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
    return np.sqrt(p * q).sum()


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
    p, q = normalize_pmfs(dist1, dist2)
    bc = bhattacharyya_coefficient_pmf(p, q)
    return bc


def bhattacharyya_distance_pmf(p, q):
    """
    Compute the Bhattacharyya distance.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    bd : float
        The Bhattacharyya distance (in bits).

    Notes
    -----
    `p` and `q` are assumed to be 1-to-1 with regards to events.
    """
    bc = bhattacharyya_coefficient_pmf(p, q)
    bc = np.clip(bc, 0.0, 1.0)
    if bc == 0.0:
        return np.inf
    return -np.log2(bc)


@unitful
def bhattacharyya_distance(dist1, dist2):
    """
    Compute the Bhattacharyya distance.

    The Bhattacharyya distance is defined as:

        D_B(P, Q) = -log_2( sum_x sqrt(P(x) Q(x)) ) = -log_2( BC(P, Q) )

    It is a non-negative quantity that equals zero if and only if P = Q.
    While it does not satisfy the triangle inequality, it is closely related
    to the Hellinger distance and Chernoff bound, and is widely used in
    feature selection and statistical signal detection.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    bd : float
        The Bhattacharyya distance (in bits).
    """
    p, q = normalize_pmfs(dist1, dist2)
    bd = bhattacharyya_distance_pmf(p, q)
    return bd


def bhattacharyya_angle_pmf(p, q):
    """
    Compute the Bhattacharyya angle.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    ba : float
        The Bhattacharyya angle (in radians).

    Notes
    -----
    `p` and `q` are assumed to be 1-to-1 with regards to events.
    """
    bc = bhattacharyya_coefficient_pmf(p, q)
    return np.arccos(np.clip(bc, 0.0, 1.0))


def bhattacharyya_angle(dist1, dist2):
    """
    Compute the Bhattacharyya angle.

    The Bhattacharyya angle is the angle between the two distributions once
    they are mapped to the unit sphere by the square root of their pmfs:

        theta(P, Q) = arccos( sum_x sqrt(P(x) Q(x)) ) = arccos( BC(P, Q) )

    Unlike the Bhattacharyya distance it is a true metric, satisfying the
    triangle inequality. It ranges from 0, when P = Q, to pi/2, when P and Q
    have disjoint support.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    ba : float
        The Bhattacharyya angle (in radians).
    """
    p, q = normalize_pmfs(dist1, dist2)
    ba = bhattacharyya_angle_pmf(p, q)
    return ba


def fisher_rao_distance_pmf(p, q):
    """
    Compute the Fisher-Rao distance.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    frd : float
        The Fisher-Rao distance (in radians).

    Notes
    -----
    `p` and `q` are assumed to be 1-to-1 with regards to events.
    """
    return 2 * bhattacharyya_angle_pmf(p, q)


def fisher_rao_distance(dist1, dist2):
    """
    Compute the Fisher-Rao distance.

    The square roots of the pmfs of all distributions over a fixed alphabet lie
    on the positive orthant of the unit sphere, and the Fisher information
    metric there is the round metric. The Fisher-Rao distance is the resulting
    geodesic distance, which is twice the Bhattacharyya angle:

        d_FR(P, Q) = 2 arccos( sum_x sqrt(P(x) Q(x)) )

    It is a true metric, ranging from 0, when P = Q, to pi, when P and Q have
    disjoint support.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    frd : float
        The Fisher-Rao distance (in radians).
    """
    p, q = normalize_pmfs(dist1, dist2)
    frd = fisher_rao_distance_pmf(p, q)
    return frd


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
    p, q = normalize_pmfs(dist1, dist2)
    hd = hellinger_distance_pmf(p, q)
    return hd


def matusita_distance_pmf(p, q):
    """
    Compute the Matusita distance.

    Parameters
    ----------
    p : np.ndarray
        The first pmf.
    q : np.ndarray
        The second pmf.

    Returns
    -------
    md : float
        The Matusita distance.

    Notes
    -----
    `p` and `q` are assumed to be 1-to-1 with regards to events.
    """
    return np.sqrt(2) * hellinger_distance_pmf(p, q)


def matusita_distance(dist1, dist2):
    """
    Compute the Matusita distance.

    The Matusita distance is the Euclidean distance between the square roots of
    the two pmfs:

        d_M(P, Q) = sqrt( sum_x ( sqrt(P(x)) - sqrt(Q(x)) )^2 )
                  = sqrt( 2 ( 1 - BC(P, Q) ) )

    It is a true metric, and is the ``dit`` Hellinger distance scaled by
    sqrt(2); it ranges from 0, when P = Q, to sqrt(2), when P and Q have
    disjoint support.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    md : float
        The Matusita distance.
    """
    p, q = normalize_pmfs(dist1, dist2)
    md = matusita_distance_pmf(p, q)
    return md


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
        return np.log2((p**alpha * q ** (1 - alpha)).sum())

    res = minimize_scalar(fun=func, bounds=(0, 1), method="bounded")

    if not -1e-8 <= res.x <= 1 + 1e-8:  # pragma: no cover
        msg = "Appropriate optima could not be found."
        raise OptimizationException(msg)

    ci = -func(res.x)
    # sometimes things are very slightly negative due to optimization fuzziness.
    # since this can throw off some inequalities, we set to zero in this case.
    if ci < 0:  # pragma: no cover
        ci = 0

    return ci


@unitful
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
    p, q = normalize_pmfs(dist1, dist2)
    ci = chernoff_information_pmf(p, q)
    return ci
