"""
The perplexity of a distribution.
"""

from ..shannon import conditional_entropy, entropy
from ..utils.misc import flatten

__all__ = ("perplexity",)


def perplexity(dist, rvs=None, crvs=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution from which the perplexity is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the perplexity.
        If None, then the perpelxity is calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.

    Returns
    -------
    P : float
        The perplexity.
    """
    base = dist.get_base(numerical=True) if dist.is_log() else 2

    if crvs is None:
        crvs = []
    if rvs is None:
        if not crvs:
            return base ** entropy(dist)
        rvs = range(dist.outcome_length())
    rvs = set(flatten(rvs))

    return base ** conditional_entropy(dist, rvs, crvs)
