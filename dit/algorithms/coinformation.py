"""
The co-information aka the multivariate mututal information.
"""

from iterutils import powerset

from ..exceptions import ditException
from .shannon import conditional_entropy as H

def coinformation(dist, rvs=None, crvs=None, rv_names=None):
    """
    Calculates the coinformation.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the coinformation is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the coinformation
        between. If None, then the coinformation is calculated over all random
        variables.
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
    I : float
        The coinformation.

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
        msg = "The coinformation is applicable to joint distributions."
        raise ditException(msg)

    def entropy(rvs, dist=dist, crvs=crvs, rv_names=rv_names):
        return H(dist, set().union(*rvs), crvs, rv_names)

    I = sum( (-1)**(len(Xs)+1) * entropy(Xs) for Xs in powerset(rvs) )

    return I
