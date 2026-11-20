"""
The Rényi dual total correlation.
"""

from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..utils import unitful
from .renyi_total_correlation import renyi_total_correlation

__all__ = ("renyi_dual_total_correlation",)


@unitful
def renyi_dual_total_correlation(dist, order, rvs=None, crvs=None):
    """
    Computes the Rényi dual total correlation of order `order`.

    The Rényi dual total correlation generalizes the dual total correlation (or
    binding information) using the Rényi entropy of order `order`:

        B_alpha(X_0, ..., X_{n-1}) = sum_i H_alpha(X_{-i}) - (n - 1) H_alpha(X_0, ..., X_{n-1})

    where X_{-i} is every variable except X_i.

    When `order=1`, this reduces to the classical Shannon-based dual total
    correlation.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the Rényi dual total correlation is
        calculated.
    order : float >= 0
        The order of the Rényi entropy.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the Rényi dual total correlation. If None,
        then it is calculated over all random variables, which is equivalent to
        passing `rvs=dist.rvs`.
    crvs : list, None
        Conditioning variables are not currently supported for the Rényi dual
        total correlation. Must be None or empty.

    Returns
    -------
    B_a : float
        The Rényi dual total correlation.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution, if `rvs` contains
        non-existent random variables, or if `crvs` is specified.
    ValueError
        Raised if `order` is not a non-negative float.

    Notes
    -----
    The value is accumulated from Rényi total correlations using the identity

        B_alpha(X) = (n - 1) T_alpha(X) - sum_i T_alpha(X_{-i})

    which expands to the definition above. Phrased this way the computation
    needs only joint Rényi entropies. The more familiar form of the dual total
    correlation, H(X) - sum_i H(X_i | X_{-i}), would instead require a
    conditional Rényi entropy, for which there is no single agreed-upon
    definition.

    References
    ----------
    Han, T. S. (1975). Linear dependence structure of the entropy space.
    Information and Control, 29(4), 337-368.

    The Rényi form substitutes H_alpha into the Shannon definition, following
    the general prescription for Rényi generalizations of information measures
    that are linear combinations of entropies: Berta, M., Seshadreesan, K. P.,
    & Wilde, M. M. (2015). Rényi generalizations of quantum information
    measures. Physical Review A, 91(2), 022333. arXiv:1502.07977.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        msg = "Conditional Rényi dual total correlation is not currently supported."
        raise ditException(msg)

    rvs = list(rvs)
    n = len(rvs)
    if n < 2:
        return 0.0

    whole = renyi_total_correlation(dist, order, rvs=rvs)
    parts = sum(renyi_total_correlation(dist, order, rvs=rvs[:i] + rvs[i + 1 :]) for i in range(n))
    B_a = (n - 1) * whole - parts

    return B_a
