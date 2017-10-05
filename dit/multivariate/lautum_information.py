"""
The lautum (mutual backwards) information, as defined by Palomar & Verdu.
"""

from ..distconst import product_distribution
from ..divergences import kullback_leibler_divergence
from ..helpers import normalize_rvs

def lautum_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Computes the lautum information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the lautum information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the lautum information. If None, then the
        lautum information is calculated over all random variables, which is
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
    L : float
        The lautum information.

    Examples
    --------
    >>> outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
    >>> pmf = [3/16, 1/16, 1/16, 3/16, 1/16, 3/16, 3/16, 1/16]
    >>> d = dit.Distribution(outcomes, pmf)
    >>> dit.multivariate.lautum_information(d)
    0.20751874963942196
    >>> dit.multivariate.lautum_information(d, rvs=[[0], [1]])
    0.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    pd = product_distribution(dist, rvs=rvs + [crvs], rv_mode=rv_mode)
    L = kullback_leibler_divergence(pd, dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)
    return L
