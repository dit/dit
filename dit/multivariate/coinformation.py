"""
The co-information aka the multivariate mututal information.
"""

from iterutils import powerset

from ..helpers import normalize_rvs
from ..shannon import conditional_entropy as H

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

    Examples
    --------
    Let's construct a 3-variable distribution for the XOR logic gate and name
    the random variables X, Y, and Z.

    >>> d = dit.example_dists.Xor()
    >>> d.set_rv_names(['X', 'Y', 'Z'])

    The 3-way mutual information I[X:Y:Z] is:

    >>> dit.multivariate.coinformation(d, 'XYZ')
    -1.0

    Using random variable indexes instead of random variable names:

    >>> dit.multivariate.coinformation(d, [0,1,2], rv_names=False)
    -1.0

    The mutual information I[X:Z] is given by:

    >>> dit.multivariate.coinformation(d, 'XZ')
    0.0

    Conditional entropy can be calculated by passing in the conditional
    random variables. The conditional entropy I[X:Y|Z] is:

    >>> dit.multivariate.coinformation(d, 'XY', 'Z')
    1.0

    """
    rvs, crvs, rv_names = normalize_rvs(dist, rvs, crvs, rv_names)

    def entropy(rvs, dist=dist, crvs=crvs, rv_names=rv_names):
        """
        Helper function to aid in computing the entropy of subsets.
        """
        return H(dist, set().union(*rvs), crvs, rv_names)

    I = sum((-1)**(len(Xs)+1) * entropy(Xs) for Xs in powerset(rvs))

    return I
