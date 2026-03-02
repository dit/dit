"""
Compute the Gacs-Korner common information
"""

from ...algorithms import insert_meet
from ...helpers import normalize_rvs, parse_rvs
from ... import Distribution
from ...shannon import conditional_entropy as H
from ...utils import unitful

__all__ = ("gk_common_information",)


@unitful
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
        Deprecated. Kept for signature compatibility.

    Returns
    -------
    K : float
        The Gacs-Korner common information of the distribution.

    Raises
    ------
    ditException
        Raised if `rvs` or `crvs` contain non-existant random variables.

    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs)
    crvs = parse_rvs(dist, crvs)[1]

    outcomes, pmf = zip(*dist.zipped(mode="patoms"), strict=True)
    d = Distribution(outcomes, pmf)
    names = dist.get_rv_names()
    if names is not None:
        d.set_rv_names(names)

    # support_only=True restricts the sigma algebra to the support,
    # which is essential for GK common information correctness.
    d2 = insert_meet(d, -1, rvs, support_only=True)

    common = [d2.outcome_length() - 1]

    K = H(d2, common, crvs)

    return K
