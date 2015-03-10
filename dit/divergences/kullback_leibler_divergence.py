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
    """
    xh = cross_entropy(dist1, dist2, rvs, crvs, rv_mode)
    h = entropy(dist1, rvs, crvs, rv_mode)
    dkl = xh - h
    return dkl

relative_entropy = kullback_leibler_divergence
