"""
The negentropy of a distribution.
"""

import numpy as np

from ..multivariate import entropy
from ..utils.misc import flatten

__all__ = ("negentropy",)


def negentropy(dist, rvs=None):
    """
    Compute the negentropy of a distribution.

    The negentropy is the difference between the entropy of a uniform
    distribution over the same alphabet and the entropy of the distribution
    itself. It is a non-negative quantity which is zero if and only if the
    distribution is uniform, and quantifies how far the distribution is from
    uniformity.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the negentropy is calculated.
    rvs : list, None
        The indexes of the random variables used to calculate the negentropy.
        If None, then the negentropy is calculated over all random variables.

    Returns
    -------
    N : float
        The negentropy.

    Examples
    --------
    >>> d = dit.example_dists.Xor()
    >>> dit.other.negentropy(d)
    1.0

    Raises
    ------
    ditException
        Raised if `rvs` contain non-existant random variables.
    """
    base = dist.get_base(numerical=True) if dist.is_log() else 2

    rvs = list(range(dist.outcome_length())) if rvs is None else list(flatten(rvs))

    alphabet = dist.alphabet
    max_entropy = sum(np.log(len(alphabet[rv])) for rv in rvs) / np.log(base)

    return max_entropy - entropy(dist, rvs)
