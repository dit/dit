"""
Compute the Gacs-Korner common information
"""

from .lattice import meet
from ..npdist import Distribution
from .shannon import entropy as H

def common_information(dist, rvs=None, rv_names=None):
    """
    Returns the Gacs-Korner common information H[X] over the random
    variables in `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the common information is calculated. 
    rvs : list, None
        The indexes of the random variable used to calculate the entropy.
        If None, then the entropy is calculated over all random variables.
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    K : float
        The Gacs-Korner common information of the distribution.

    """
    if rvs is None:
        rvs = range(dist.outcome_length())
        rv_names = False

    outcomes, pmf = zip(*dist.zipped(mode='patoms'))
    d = Distribution(pmf, outcomes)

    d2 = meet(d, rvs, rv_names)

    K = H(d2)
    return K
