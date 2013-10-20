"""
Compute the Gacs-Korner common information
"""

from ..helpers import normalize_rvs, parse_rvs
from ..npdist import Distribution
from .lattice import insert_meet
from .shannon import conditional_entropy as H

def common_information(dist, rvs=None, crvs=None, rv_names=None):
    """
    Returns the Gacs-Korner common information K[X1:X2...] over the random
    variables in `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the common information is calculated. 
    rvs : list, None
        The indexes of the random variables for which the Gacs-Korner common
        information is to be computed. If None, then the common information is
        calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition the common information
        by. If none, than there is no conditioning.
    rv_names : bool, None
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    K : float
        The Gacs-Korner common information of the distribution.

    """
    rvs, crvs, rv_names = normalize_rvs(dist, rvs, crvs, rv_names)
    crvs = parse_rvs(dist, crvs, rv_names)[1]

    outcomes, pmf = zip(*dist.zipped(mode='patoms'))
    d = Distribution(outcomes, pmf)
    d.set_rv_names(dist.get_rv_names())

    d2 = insert_meet(d, -1, rvs, rv_names)

    common = [d2.outcome_length() - 1]

    K = H(d2, common, crvs)

    return K
