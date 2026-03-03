"""
A version of the entropy with signature common to the other multivariate
measures.
"""

from ..helpers import normalize_rvs
from ..shannon import conditional_entropy
from ..shannon import entropy as shannon_entropy
from ..utils import flatten, unitful

__all__ = ("entropy",)


@unitful
def entropy(dist, rvs=None, crvs=None):
    """
    Calculates the conditional joint entropy.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the entropy is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the entropy. If
        None, then the entropy is calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    H : float
        The entropy.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.

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

    >>> dit.multivariate.entropy(d, [0,1,2])
    2.0

    The joint entropy H[X,Z] is given by:

    >>> dit.multivariate.entropy(d, 'XZ')
    1.0

    Conditional entropy can be calculated by passing in the conditional
    random variables. The conditional entropy H[Y|X] is:

    >>> dit.multivariate.entropy(d, 'Y', 'X')
    1.0

    """
    if rvs is not None or crvs is not None:
        rvs, crvs = normalize_rvs(dist, rvs, crvs)
        rvs = list(flatten(rvs))
        H = conditional_entropy(dist, rvs, crvs)
    else:
        H = shannon_entropy(dist)

    return H
