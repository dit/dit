"""
The Kullback-Leibler divergence.
"""

from .cross_entropy import cross_entropy
from ..multivariate import entropy

__all__ = ('kullback_leibler_divergence',
           'relative_entropy',
          )

def kullback_leibler_divergence(dist1, dist2, rvs=None, crvs=None, rv_mode=None):
    """
    The Kullback-Liebler divergence between `dist1` and `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the Kullback-Leibler divergence.
    dist2 : Distribution
        The second distribution in the Kullback-Leibler divergence.
    rvs : list, None
        The indexes of the random variable used to calculate the
        Kullback-Leibler divergence between. If None, then the Kullback-Leibler
        divergence is calculated over all random variables.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    dkl : float
        The Kullback-Leibler divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.
    """
    xh = cross_entropy(dist1, dist2, rvs, crvs, rv_mode)
    h = entropy(dist1, rvs, crvs, rv_mode)
    dkl = xh - h
    return dkl

relative_entropy = kullback_leibler_divergence
