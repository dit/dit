"""
The binding information and residual entropy.
"""

from .shannon import conditional_entropy as H
from ..exceptions import ditException

def binding_information(dist, rvs=None, crvs=None, rv_names=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution from which the binding information is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the binding
        information. If None, then the binding information is calculated
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
    B : float
        The binding information

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
        msg = "The binding information is applicable to joint distributions."
        raise ditException(msg)

    others = lambda rv, rvs: set(set().union(*rvs)) - set(rv)

    one = H(dist, set().union(*rvs), crvs, rv_names)
    two = sum(H(dist, rv, others(rv, rvs).union(crvs), rv_names) for rv in rvs)
    B = one - two

    return B


def residual_entropy(dist, rvs=None, crvs=None, rv_names=None):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution from which the residual entropy is calculated.
    rvs : list, None
        The indexes of the random variable used to calculate the residual
        entropy. If None, then the total correlation is calculated
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
    R : float
        The residual entropy

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
        msg = "The residual entropy is applicable to joint distributions."
        raise ditException(msg)

    others = lambda rv, rvs: set(set().union(*rvs)) - set(rv)

    R = sum(H(dist, rv, others(rv, rvs).union(crvs), rv_names) for rv in rvs)

    return R
