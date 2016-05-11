"""
Compute the Gacs-Korner common information
"""

from ..helpers import normalize_rvs, parse_rvs
from ..npdist import Distribution
from ..algorithms import insert_meet
from ..shannon import conditional_entropy as H

def gk_common_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Calculates the Gacs-Korner common information K[X1:X2...] over the random
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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    K : float
        The Gacs-Korner common information of the distribution.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.

    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
    crvs = parse_rvs(dist, crvs, rv_mode)[1]

    outcomes, pmf = zip(*dist.zipped(mode='patoms'))
    # The GK-common information is sensitive to zeros in the sample space.
    # Here, we make sure to remove them.
    d = Distribution(outcomes, pmf, sample_space=outcomes)
    d.set_rv_names(dist.get_rv_names())

    d2 = insert_meet(d, -1, rvs, rv_mode=rv_mode)

    common = [d2.outcome_length() - 1]

    K = H(d2, common, crvs)

    return K
