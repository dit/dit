"""

"""

from ..total_correlation import total_correlation
from ..dual_total_correlation import dual_total_correlation
from ..caekl_mutual_information import caekl_mutual_information

__all__ = [
    'lower_intrinsic_mutual_information',
    'upper_intrinsic_total_correlation',
    'upper_intrinsic_dual_total_correlation',
    'upper_intrinsic_caekl_mutual_information',
]


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
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    lb : float
        A lower-bound on the secret key agreement rate.
    """
    a = total_correlation(dist, [X, Y], rv_mode=rv_mode)
    b = total_correlation(dist, [X, Z], rv_mode=rv_mode)
    return max(0.0, a-b)


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
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    lb : float
        A lower-bound on the secret key agreement rate.
    """
    a = lower_intrinsic_mutual_information_directed(dist, rvs[0], rvs[1], crvs, rv_mode=None)
    b = lower_intrinsic_mutual_information_directed(dist, rvs[1], rvs[0], crvs, rv_mode=None)
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
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    ub : float
        A lower-bound on the secret key agreement rate.
    """
    a = total_correlation(dist, rvs, rv_mode=rv_mode)
    b = total_correlation(dist, rvs, crvs, rv_mode=rv_mode)
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
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    ub : float
        A lower-bound on the secret key agreement rate.
    """
    a = dual_total_correlation(dist, rvs, rv_mode=rv_mode)
    b = dual_total_correlation(dist, rvs, crvs, rv_mode=rv_mode)
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
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    ub : float
        A lower-bound on the secret key agreement rate.
    """
    a = caekl_mutual_information(dist, rvs, rv_mode=rv_mode)
    b = caekl_mutual_information(dist, rvs, crvs, rv_mode=rv_mode)
    return min([a, b])
