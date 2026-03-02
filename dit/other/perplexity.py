"""
The perplexity of a distribution.
"""

from ..shannon import conditional_entropy, entropy
from ..utils.misc import flatten

__all__ = ("perplexity",)


def perplexity(dist, rvs=None, crvs=None, rv_mode=None):
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
    rv_mode : str, None
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    P : float
        The perplexity.
    """
    base = dist.get_base(numerical=True) if dist.is_log() else 2

    if rvs is not None:
        rvs = set(flatten(rvs))
    if crvs is None:
        crvs = []
    if rvs is None:
        return base ** entropy(dist)

    return base ** conditional_entropy(dist, rvs, crvs)
