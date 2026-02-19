"""
The O-information, as defined by Rosas et al.
"""

from .dual_total_correlation import dual_total_correlation
from .total_correlation import total_correlation

__all__ = (
    'o_information',
)


def o_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Computes the O-information, defined as the total correlation minus the dual
    total correlation.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the o-information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the o-information. If None, then the
        o-information is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    O : float
        The o-information.

    Examples
    --------
    >>> d = dit.example_dists.n_mod_m(5, 2)
    >>> dit.multivariate.o_information(d)
    3.0
    >>> dit.multivariate.o_information(d, rvs=[[0], [1], [3], [4]], [2])
    -2.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    t = total_correlation(dist=dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)
    b = dual_total_correlation(dist=dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)
    return t - b
