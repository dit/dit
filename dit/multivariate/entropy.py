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

    Examples
    --------
    Let's construct a 3-variable distribution for the XOR logic gate and name
    the random variables X, Y, and Z.

    >>> d = dit.example_dists.Xor()
    >>> d.set_rv_names(['X', 'Y', 'Z'])

    The joint entropy of H[X,Y,Z] is:

    >>> dit.multivariate.entropy(d, 'XYZ')
    2.0

    We can do this using random variables indexes too.

    >>> dit.multivariate.entropy(d, [0,1,2], rv_mode='indexes')
    2.0

    The joint entropy H[X,Z] is given by:

    >>> dit.multivariate.entropy(d, 'XZ')
    1.0

    Conditional entropy can be calculated by passing in the conditional
    random variables. The conditional entropy H[Y|X] is:

    >>> dit.multivariate.entropy(d, 'Y', 'X')
    1.0

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
