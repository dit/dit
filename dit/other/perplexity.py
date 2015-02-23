"""
The perplexity of a distribution.
"""

from ..helpers import RV_MODES
from ..shannon import conditional_entropy, entropy
from ..utils.misc import flatten

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
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    P : float
        The perplexity.
    """

    base = dist.get_base(numerical=True) if dist.is_log() else 2

    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = list(range(dist.outcome_length()))
            rv_mode = RV_MODES.INDICES
        else:
            # this will allow inputs of the form [0, 1, 2] or [[0, 1], [2]],
            # allowing uniform behavior with the mutual information like
            # measures.
            rvs = set(flatten(rvs))
        if crvs is None:
            crvs = []
    else:
        return base**entropy(dist)

    return base**conditional_entropy(dist, rvs, crvs, rv_mode=rv_mode)
