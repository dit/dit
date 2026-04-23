"""
The Kullback-Leibler divergence.
"""

from ..multivariate.entropy import entropy
from .cross_entropy import cross_entropy

__all__ = (
    "kullback_leibler_divergence",
    "relative_entropy",
)


def kullback_leibler_divergence(dist1, dist2, rvs=None, crvs=None):
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
    xh = cross_entropy(dist1, dist2, rvs, crvs)
    h = entropy(dist1, rvs, crvs)
    dkl = xh - h
    return dkl


relative_entropy = kullback_leibler_divergence
