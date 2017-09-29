"""
The co-information aka the multivariate mututal information.
"""

from ..helpers import normalize_rvs
from ..shannon import conditional_entropy as H
from ..utils import powerset

def coinformation(dist, rvs=None, crvs=None, rv_mode=None):
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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    I : float
        The coinformation.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.

    Examples
    --------
    Let's construct a 3-variable distribution for the XOR logic gate and name
    the random variables X, Y, and Z.

    >>> d = dit.example_dists.Xor()
    >>> d.set_rv_names(['X', 'Y', 'Z'])

    To calculate coinformations, recall that `rvs` specifies which groups of
    random variables are involved. For example, the 3-way mutual information
    I[X:Y:Z] is calculated as:

    >>> dit.multivariate.coinformation(d, ['X', 'Y', 'Z'])
    -1.0

    It is a quirk of strings that each element of a string is also an iterable.
    So an equivalent way to calculate the 3-way mutual information I[X:Y:Z] is:

    >>> dit.multivariate.coinformation(d, 'XYZ')
    -1.0

    The reason this works is that list('XYZ') == ['X', 'Y', 'Z']. If we want
    to use random variable indexes, we need to have explicit groupings:

    >>> dit.multivariate.coinformation(d, [[0], [1], [2]], rv_mode='indexes')
    -1.0



    To calculate the mutual information I[X, Y : Z], we use explicit groups:

    >>> dit.multivariate.coinformation(d, ['XY', 'Z'])

    Using indexes, this looks like:

    >>> dit.multivariate.coinformation(d, [[0, 1], [2]], rv_mode='indexes')



    The mutual information I[X:Z] is given by:

    >>> dit.multivariate.coinformation(d, 'XZ')
    0.0

    Equivalently,

    >>> dit.multivariate.coinformation(d, ['X', 'Z'])
    0.0

    Using indexes, this becomes:

    >>> dit.multivariate.coinformation(d, [[0], [2]])
    0.0



    Conditional mutual informations can be calculated by passing in the
    conditional random variables. The conditional entropy I[X:Y|Z] is:

    >>> dit.multivariate.coinformation(d, 'XY', 'Z')
    1.0

    Using indexes, this becomes:

    >>> rvs = [[0], [1]]
    >>> crvs = [[2]] # broken
    >>> dit.multivariate.coinformation(d, rvs, crvs, rv_mode='indexes')
    1.0

    For the conditional random variables, groupings have no effect, so you
    can also obtain this as:

    >>> rvs = [[0], [1]]
    >>> crvs = [2]
    >>> dit.multivariate.coinformation(d, rvs, crvs, rv_mode='indexes')
    1.0



    Finally, note that entropy can also be calculated. The entropy H[Z|XY]
    is obtained as:

    >>> rvs = [[2]]
    >>> crvs = [[0], [1]] # broken
    >>> dit.multivariate.coinformation(d, rvs, crvs, rv_mode='indexes')
    0.0

    >>> crvs = [[0, 1]] # broken
    >>> dit.multivariate.coinformation(d, rvs, crvs, rv_mode='indexes')
    0.0

    >>> crvs = [0, 1]
    >>> dit.multivariate.coinformation(d, rvs, crvs, rv_mode='indexes')
    0.0

    >>> rvs = 'Z'
    >>> crvs = 'XY'
    >>> dit.multivariate.coinformation(d, rvs, crvs, rv_mode='indexes')
    0.0

    Note that [[0], [1]] says to condition on two groups. But conditioning
    is a flat operation and doesn't respect the groups, so it is equal to
    a single group of 2 random variables: [[0, 1]]. With random variable
    names 'XY' is acceptable because list('XY') = ['X', 'Y'], which is
    species two singleton groups. By the previous argument, this is will
    be treated the same as ['XY'].

    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    def entropy(rvs, dist=dist, crvs=crvs, rv_mode=rv_mode):
        """
        Helper function to aid in computing the entropy of subsets.
        """
        return H(dist, set().union(*rvs), crvs, rv_mode=rv_mode)

    I = sum((-1)**(len(Xs)+1) * entropy(Xs) for Xs in powerset(rvs))

    return I
