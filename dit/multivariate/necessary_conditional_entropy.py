"""

"""

from .entropy import entropy
from ..algorithms import insert_mss
from ..helpers import normalize_rvs
from ..utils import flatten

def necessary_conditional_entropy(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Calculates the necessary conditional entropy :math:`\H[X \dagger Y]`.
    This is the entropy of the minimal sufficient statistic of X about Y, given
    Y.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the necessary conditional entropy is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the necessary
        conditional entropy. If None, then the entropy is calculated over 
        all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    H : float
        The necessary conditional entropy.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.

    Example
    -------

    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
    rvs = list(flatten(rvs))
    d = insert_mss(dist, -1, rvs, about=crvs, rv_mode=rv_mode)
    H = entropy(d, [dist.outcome_length()], crvs, rv_mode)
    return H
