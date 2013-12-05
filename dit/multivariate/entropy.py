"""
A version of the entropy with signature common to the other multivariate
measures.
"""

from ..shannon import conditional_entropy, entropy as shannon_entropy

def entropy(dist, rvs=None, crvs=None, rv_names=None):
    """
    Compute the conditional joint entropy.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the entropy is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the entropy. If
        None, then the entropy is calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    H : float
        The entropy.
    """
    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = list(range(dist.outcome_length()))
            rv_names = False
        if crvs is None:
            crvs = []
    else:
        return shannon_entropy(dist)

    return conditional_entropy(dist, rvs, crvs, rv_names)
