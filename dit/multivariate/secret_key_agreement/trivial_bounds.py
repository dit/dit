"""
Some trivial bounds to the two-way secret key agreement problem.
"""

from ..caekl_mutual_information import caekl_mutual_information
from ..dual_total_correlation import dual_total_correlation
from ..total_correlation import total_correlation

__all__ = (
    "lower_intrinsic_mutual_information",
    "upper_intrinsic_total_correlation",
    "upper_intrinsic_dual_total_correlation",
    "upper_intrinsic_caekl_mutual_information",
)


def lower_intrinsic_mutual_information_directed(dist, X, Y, Z, rv_mode=None):
    """
    A lower bound on the secrecy capacity:
        I[X:Y] - I[X:Z]

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the bound for.
    X : iterable
        The variables representing Alice.
    Y : iterable
        The variables representing Bob.
    Z : iterable
        The variables representing Eve.
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    lb : float
        A lower-bound on the secret key agreement rate.
    """
    a = total_correlation(dist, [X, Y])
    b = total_correlation(dist, [X, Z])
    return max(0.0, a - b)


def lower_intrinsic_mutual_information(dist, rvs, crvs, rv_mode=None):
    """
    Compute a trivial lower-bound on the secret key agreement rate.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the bound for.
    rvs : iterable of iterables, len(rvs) == 2
        The variables representing the agents to agree upon a key.
    crvs : iterable
        The variable representing the eavesdropper.
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    lb : float
        A lower-bound on the secret key agreement rate.
    """
    a = lower_intrinsic_mutual_information_directed(dist, rvs[0], rvs[1], crvs)
    b = lower_intrinsic_mutual_information_directed(dist, rvs[1], rvs[0], crvs)
    return max([a, b])


def upper_intrinsic_total_correlation(dist, rvs, crvs, rv_mode=None):
    """
    Compute a trivial upper-bound on the secret key agreement rate.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the bound for.
    rvs : iterable of iterables
        The variables representing the agents to agree upon a key.
    crvs : iterable
        The variable representing the eavesdropper.
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    ub : float
        A lower-bound on the secret key agreement rate.
    """
    a = total_correlation(dist, rvs)
    b = total_correlation(dist, rvs, crvs)
    return min([a, b])


def upper_intrinsic_dual_total_correlation(dist, rvs, crvs, rv_mode=None):
    """
    Compute a trivial upper-bound on the secret key agreement rate.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the bound for.
    rvs : iterable of iterables
        The variables representing the agents to agree upon a key.
    crvs : iterable
        The variable representing the eavesdropper.
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    ub : float
        A lower-bound on the secret key agreement rate.
    """
    a = dual_total_correlation(dist, rvs)
    b = dual_total_correlation(dist, rvs, crvs)
    return min([a, b])


def upper_intrinsic_caekl_mutual_information(dist, rvs, crvs, rv_mode=None):
    """
    Compute a trivial upper-bound on the secret key agreement rate.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the bound for.
    rvs : iterable of iterables
        The variables representing the agents to agree upon a key.
    crvs : iterable
        The variable representing the eavesdropper.
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    ub : float
        A lower-bound on the secret key agreement rate.
    """
    a = caekl_mutual_information(dist, rvs)
    b = caekl_mutual_information(dist, rvs, crvs)
    return min([a, b])
