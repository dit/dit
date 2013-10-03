"""
The co-information.
"""

from iterutils import flatten, powerset

from .shannon import conditional_entropy as H

def coinformation(dist, rvs=None, crvs=None, rv_names=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution from which the total correlation is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the total
        correlation. If None, then the total correlation is calculated
        over all random variables.
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
            rvs = [ [i] for i in range(dist.outcome_length()) ]
            rv_names = False
        if crvs is None:
            crvs = []
    else:
        msg = "The total correlation is applicable to joint distributions."
        raise ditException(msg)

    I = sum( (-1)**(len(Xs)+1)*H(dist, flatten(Xs), crvs, rv_names) for Xs in powerset(rvs) )

    return I