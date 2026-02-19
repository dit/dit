"""
The secrecy capacity is the rate at which X and Y can agree upon a secret key
while Z eavesdrops, and X controlls the channel p(YZ|X).

Given a fixed joint p(XYZ), the secrecy capacity is a lower bound on the one-way
secret key agreement rate since arbitrary input to the channel p(YZ|X) can be
emulated through proper communication on the part of X.
"""

from .one_way_skar import OneWaySKAR
from ...utils import unitful
from .._backend import _make_backend_subclass


__all__ = (
    'secrecy_capacity',
)


class SecrecyCapacity(OneWaySKAR):
    """
    Compute:
        max_{U - X - YZ} I[U:Y] - I[U:Z]
    """

    def _get_v_bound(self):
        """
        Make V a constant.
        """
        return 1


@unitful
def secrecy_capacity(dist, X, Y, Z, rv_mode=None, niter=None, bound_u=None,
                     backend='numpy'):
    """
    The rate at which X and Y can agree upon a key over the channel p(YZ|X)
    while Z eavesdrops, and no public communication.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    X : iterable
        The indices to consider as the X variable, Alice.
    Y : iterable
        The indices to consider as the Y variable, Bob.
    Z : iterable
        The indices to consider as the Z variable, Eve.
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
    actual_cls = _make_backend_subclass(SecrecyCapacity, backend)
    sc = actual_cls(dist, X, Y, Z, rv_mode=rv_mode, bound_u=bound_u)
    sc.optimize(niter=niter)
    value = -sc.objective(sc._optima)

    return value
