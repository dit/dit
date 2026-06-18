"""
Sibson (alpha) mutual information and conditional variants.

Unconditional Sibson MI generalizes Shannon mutual information (order 1) and
maximal leakage (order infinity).  Conditional forms follow Esposito et al.
(2102.00720) and Wu et al. (2005.06033).
"""

import numpy as np

from ..shannon import mutual_information
from ..utils import flatten

__all__ = (
    "maximal_leakage",
    "sibson_conditional_mutual_information_y_given_z",
    "sibson_conditional_mutual_information_z",
    "sibson_mutual_information",
    "sibson_mutual_information_pmf",
)


def _validate_order(order):
    if order <= 0:
        msg = "`order` must be a positive real number"
        raise ValueError(msg)


def _joint_pmf(dist, rvs_X, rvs_Y):
    """Return joint P(X,Y) as a 2-D array with axis 0 = X, axis 1 = Y."""
    joint = dist.coalesce([rvs_X, rvs_Y])
    joint.make_dense()
    shape = [len(a) for a in joint.alphabet]
    return joint.pmf.reshape(shape)


def _sibson_mi_from_joint_pmf(p_xy, order):
    """
    Compute Sibson MI from joint PMF ``p_xy`` (axis 0 = X, axis 1 = Y).

    I_alpha(X;Y) = alpha/(alpha-1) log2 sum_y (sum_x P(x) P(y|x)^alpha)^(1/alpha)
    """
    _validate_order(order)

    p_x = p_xy.sum(axis=1)
    p_ygx = np.zeros_like(p_xy)
    np.divide(p_xy, p_x[:, None], out=p_ygx, where=p_x[:, None] > 0)

    if order == 1:
        p_y = p_xy.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                p_xy > 0,
                p_xy * np.log2(p_xy / (p_x[:, None] * p_y)),
                0.0,
            )
        return float(terms.sum())

    if order == np.inf:
        mask = p_x > 0
        if not np.any(mask):
            return 0.0
        return float(np.log2(np.max(p_ygx[mask], axis=0).sum()))

    inner = (p_x[:, None] * p_ygx**order).sum(axis=0)
    return float((order / (order - 1)) * np.log2(np.sum(inner ** (1 / order))))


def sibson_mutual_information_pmf(p_xy, order):
    """
    Compute the Sibson mutual information of order ``order``.

    Parameters
    ----------
    p_xy : array_like, shape (n_x, n_y)
        Joint PMF with axis 0 indexing ``X`` and axis 1 indexing ``Y``.
    order : float > 0
        The order ``alpha``.  Use ``1`` for Shannon MI and ``numpy.inf`` for
        maximal leakage.

    Returns
    -------
    I_a : float
        Sibson mutual information in bits.

    Raises
    ------
    ValueError
        If ``order`` is not positive.
    """
    return _sibson_mi_from_joint_pmf(np.asarray(p_xy, dtype=float), order)


def sibson_mutual_information(dist, rvs_X, rvs_Y, order):
    """
    Compute the Sibson mutual information of order ``order``.

    This is asymmetric in ``(X, Y)``: ``rvs_X`` is the source variable whose
    marginal appears in the definition (Wu et al., Verdu).

    Parameters
    ----------
    dist : Distribution
        The joint distribution.
    rvs_X : list
        Indexes of the random variables defining ``X``.
    rvs_Y : list
        Indexes of the random variables defining ``Y``.
    order : float > 0
        The order ``alpha``.

    Returns
    -------
    I_a : float
        Sibson mutual information in bits.

    Raises
    ------
    ValueError
        If ``order`` is not positive.
    """
    _validate_order(order)

    if order == 1:
        return mutual_information(dist, rvs_X, rvs_Y)

    p_xy = _joint_pmf(dist, rvs_X, rvs_Y)
    return _sibson_mi_from_joint_pmf(p_xy, order)


def maximal_leakage(dist, rvs_X, rvs_Y):
    """
    Maximal leakage from ``X`` to ``Y``.

    Equivalent to Sibson mutual information of order infinity.

    Parameters
    ----------
    dist : Distribution
        The joint distribution.
    rvs_X : list
        Indexes of the source random variables.
    rvs_Y : list
        Indexes of the observed random variables.

    Returns
    -------
    L : float
        Maximal leakage in bits.
    """
    return sibson_mutual_information(dist, rvs_X, rvs_Y, np.inf)


def _conditional_y_given_z_pmf(p_xyz, order):
    """
    I^Y|Z_alpha from a 3-D joint PMF (axis 0=X, 1=Y, 2=Z).

    min_{Q_{Y|Z}} D_alpha(P_{XYZ} || P_{X|Z} Q_{Y|Z} P_Z) decomposes over z as a
    weighted sum of unconditional Sibson terms on each P_{XY|z}.
    """
    _validate_order(order)

    p_z = p_xyz.sum(axis=(0, 1))
    n_z = p_xyz.shape[2]

    total = 0.0
    for z in range(n_z):
        if p_z[z] == 0:
            continue
        p_xy_z = p_xyz[:, :, z]
        total += p_z[z] * _sibson_mi_from_joint_pmf(p_xy_z, order)

    return float(total)


def _conditional_z_pmf(p_xyz, order):
    """
    I^Z_alpha from a 3-D joint PMF (axis 0=X, 1=Y, 2=Z).

    alpha/(alpha-1) log2 sum_z P(z) (sum_{x,y} P(x,y|z)^alpha P(x|z)^{1-alpha}
    P(y|z)^{1-alpha})^{1/alpha}
    """
    _validate_order(order)

    p_z = p_xyz.sum(axis=(0, 1))
    n_z = p_xyz.shape[2]

    if order == 1:
        total = 0.0
        for z in range(n_z):
            if p_z[z] == 0:
                continue
            p_xy_z = p_xyz[:, :, z]
            p_x_z = p_xy_z.sum(axis=1)
            p_y_z = p_xy_z.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(
                    p_xy_z > 0,
                    p_xy_z * np.log2(p_xy_z * p_z[z] / (p_x_z[:, None] * p_y_z)),
                    0.0,
                )
            total += terms.sum()
        return float(total)

    if order == np.inf:
        total = 0.0
        for z in range(n_z):
            if p_z[z] == 0:
                continue
            p_xy_z = p_xyz[:, :, z]
            p_x_z = p_xy_z.sum(axis=1)
            p_y_z = p_xy_z.sum(axis=0)
            ess_sup = 0.0
            for x in range(p_xy_z.shape[0]):
                for y in range(p_xy_z.shape[1]):
                    if p_xy_z[x, y] == 0:
                        continue
                    if p_x_z[x] > 0 and p_y_z[y] > 0:
                        ratio = p_xy_z[x, y] / (p_x_z[x] * p_y_z[y])
                        ess_sup = max(ess_sup, ratio)
            total += p_z[z] * ess_sup
        return float(np.log2(total))

    inner_sum = 0.0
    for z in range(n_z):
        if p_z[z] == 0:
            continue
        p_xy_z = p_xyz[:, :, z]
        p_x_z = p_xy_z.sum(axis=1)
        p_y_z = p_xy_z.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.sum(
                np.where(
                    p_xy_z > 0,
                    p_xy_z**order
                    * np.where(p_x_z[:, None] > 0, p_x_z[:, None], 1.0) ** (1 - order)
                    * np.where(p_y_z > 0, p_y_z, 1.0) ** (1 - order),
                    0.0,
                )
            )
        inner_sum += p_z[z] * term ** (1 / order)

    return float((order / (order - 1)) * np.log2(inner_sum))


def _joint_pmf_xyz(dist, rvs_X, rvs_Y, rvs_Z):
    joint = dist.coalesce([rvs_X, rvs_Y, rvs_Z])
    joint.make_dense()
    shape = [len(a) for a in joint.alphabet]
    return joint.pmf.reshape(shape)


def sibson_conditional_mutual_information_y_given_z(dist, rvs_X, rvs_Y, rvs_Z, order):
    """
    Conditional Sibson MI minimizing over ``Q_{Y|Z}`` (Esposito et al., Def. 3).

    Reduces to unconditional Sibson MI when ``Z`` is constant (Esposito et al.,
    Def. 3).

    Parameters
    ----------
    dist : Distribution
        The joint distribution over ``X``, ``Y``, and ``Z``.
    rvs_X, rvs_Y, rvs_Z : list
        Indexes defining each variable group.
    order : float > 0
        The order ``alpha``.

    Returns
    -------
    I_a : float
        Conditional Sibson mutual information in bits.
    """
    _validate_order(order)
    rvs_X = list(flatten(rvs_X))
    rvs_Y = list(flatten(rvs_Y))
    rvs_Z = list(flatten(rvs_Z))

    p_xyz = _joint_pmf_xyz(dist, rvs_X, rvs_Y, rvs_Z)
    return _conditional_y_given_z_pmf(p_xyz, order)


def sibson_conditional_mutual_information_z(dist, rvs_X, rvs_Y, rvs_Z, order):
    """
    Conditional Sibson MI minimizing over ``Q_Z`` (Esposito et al., Def. 4).

    Symmetric in ``X`` and ``Y``.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over ``X``, ``Y``, and ``Z``.
    rvs_X, rvs_Y, rvs_Z : list
        Indexes defining each variable group.
    order : float > 0
        The order ``alpha``.

    Returns
    -------
    I_a : float
        Conditional Sibson mutual information in bits.
    """
    _validate_order(order)
    rvs_X = list(flatten(rvs_X))
    rvs_Y = list(flatten(rvs_Y))
    rvs_Z = list(flatten(rvs_Z))

    p_xyz = _joint_pmf_xyz(dist, rvs_X, rvs_Y, rvs_Z)
    return _conditional_z_pmf(p_xyz, order)
