"""
The S-information, as defined by Rosas et al.
"""

from .dual_total_correlation import dual_total_correlation
from .total_correlation import total_correlation

__all__ = ("s_information",)


def s_information(dist, rvs=None, crvs=None):
    """
    Computes the S-information, defined as the sum of the total correlation and
    the dual total correlation.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the s-information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the s-information. If None, then the
        s-information is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    S : float
        The s-information.

    Examples
    --------
    >>> d = dit.example_dists.n_mod_m(5, 2)
    >>> dit.multivariate.s_information(d)
    5.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    t = total_correlation(dist=dist, rvs=rvs, crvs=crvs)
    b = dual_total_correlation(dist=dist, rvs=rvs, crvs=crvs)
    return t + b
