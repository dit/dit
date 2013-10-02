"""
The total correlation, aka the multiinformation or the integration.
"""

from ..exceptions import ditException
from .shannon import entropy as H

def total_correlation(dist, rvs=None, rv_names=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution from which the total correlation is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the entropy.
        If None, then the total correlation is calculated over all random
        variables.
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    T : float
        The total correlation

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution.
    """
    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = range(dist.outcome_length())
            rv_names = False

        d = dist.marginal(rvs, rv_names=rv_names)
    else:
        msg = "The total correlation is applicable to joint distributions."
        raise ditException(msg)

    marginals = [ d.marginal([i]) for i in range(d.outcome_length()) ]

    T = sum( H(m) for m in marginals ) - H(d)

    return T