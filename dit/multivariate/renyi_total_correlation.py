"""
The Rényi total correlation.
"""

from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..other import renyi_entropy
from ..utils import unitful

__all__ = ("renyi_total_correlation",)


@unitful
def renyi_total_correlation(dist, order, rvs=None, crvs=None):
    """
    Computes the Rényi total correlation of order `order`.

    The Rényi total correlation generalizes Watanabe's total correlation (or
    multi-information) using the Rényi entropy of order `order`:

        TC_alpha(X_1, ..., X_n) = sum_{i=1}^n H_alpha(X_i) - H_alpha(X_1, ..., X_n)

    When `order=1`, this reduces to the classical Shannon-based total correlation.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the Rényi total correlation is calculated.
    order : float >= 0
        The order of the Rényi entropy.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the Rényi total correlation. If None, then it
        is calculated over all random variables, which is equivalent to passing
        `rvs=dist.rvs`.
    crvs : list, None
        Conditioning variables are not currently supported for Rényi total
        correlation. Must be None or empty.

    Returns
    -------
    T_a : float
        The Rényi total correlation.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution, if `rvs` contains
        non-existent random variables, or if `crvs` is specified.
    ValueError
        Raised if `order` is not a non-negative float.

    References
    ----------
    Hu, G. (2014). Rényi Information Measures for Multi-terminal Systems.
    arXiv:1403.5731.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        msg = "Conditional Rényi total correlation is not currently supported."
        raise ditException(msg)

    one = sum(renyi_entropy(dist, order, rv) for rv in rvs)
    two = renyi_entropy(dist, order, set().union(*rvs))
    T_a = one - two

    return T_a
