"""
Two lower bounds on the two-way secret key agreement rate.
"""

from ...utils import unitful
from .one_way_skar import one_way_skar
from .secrecy_capacity import secrecy_capacity

__all__ = (
    'necessary_intrinsic_mutual_information',
    'secrecy_capacity_skar',
)


@unitful
def secrecy_capacity_skar(dist, rvs=None, crvs=None, rv_mode=None,
                          niter=None, bound_u=None, backend='numpy'):
    """
    The rate at which X and Y can agree upon a key with Z eavesdropping,
    and no public communication.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The indices of the random variables agreeing upon a secret key.
    crvs : iterable
        The indices of the eavesdropper.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    niter : int, None
        The number of hops to perform during optimization.
    bound_u : int, None
        The bound to use on the size of the variable U. If none, use the
        theoretical bound of |X|.
    backend : str
        The optimization backend. One of ``'numpy'`` (default),
        ``'jax'``, or ``'torch'``.

    Returns
    -------
    sc : float
        The secrecy capacity.
    """
    a = secrecy_capacity(dist, rvs[0], rvs[1], crvs, rv_mode=rv_mode,
                         niter=niter, bound_u=bound_u, backend=backend)
    b = secrecy_capacity(dist, rvs[1], rvs[0], crvs, rv_mode=rv_mode,
                         niter=niter, bound_u=bound_u, backend=backend)
    return max([a, b])


@unitful
def necessary_intrinsic_mutual_information(dist, rvs, crvs, rv_mode=None,
                                           niter=None, bound_u=None,
                                           bound_v=None, backend='numpy'):
    """
    Compute a non-trivial lower bound on secret key agreement rate.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The indices of the random variables agreeing upon a secret key.
    crvs : iterable
        The indices of the eavesdropper.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    niter : int, None
        The number of hops to perform during optimization.
    bound_u : int, None
        The bound to use on the size of the variable U. If none, use the
        theoretical bound of |X|.
    bound_v : int, None
        The bound to use on the size of the variable V. If none, use the
        theoretical bound of |X|^2.
    backend : str
        The optimization backend. One of ``'numpy'`` (default),
        ``'jax'``, or ``'torch'``.

    Returns
    -------
    nimi : float
        The necessary intrinsic mutual information.
    """
    a = one_way_skar(dist, rvs[0], rvs[1], crvs, rv_mode=rv_mode,
                     niter=niter, bound_u=bound_u, bound_v=bound_v,
                     backend=backend)
    b = one_way_skar(dist, rvs[1], rvs[0], crvs, rv_mode=rv_mode,
                     niter=niter, bound_u=bound_u, bound_v=bound_v,
                     backend=backend)

    return max([a, b])
