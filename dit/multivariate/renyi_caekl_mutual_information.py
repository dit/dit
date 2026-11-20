"""
The Rényi CAEKL mutual information.
"""

from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..utils import partitions, unitful
from .renyi_total_correlation import renyi_total_correlation

__all__ = ("renyi_caekl_mutual_information",)


@unitful
def renyi_caekl_mutual_information(dist, order, rvs=None, crvs=None):
    """
    Computes the Rényi Chan-AlBashabsheh-Ebrahimi-Kaced-Liu mutual information
    of order `order`.

    The CAEKL mutual information is the smallest per-partition total
    correlation, minimized over non-trivial partitions of the variables::

        J(X_0, ..., X_{n-1}) = min_P T(X_C : C in P) / (|P| - 1)

    The Rényi version keeps that definition and swaps the total correlation for
    the Rényi total correlation of order `order`::

        J_alpha(X_0, ..., X_{n-1}) = min_P T_alpha(X_C : C in P) / (|P| - 1)

    When `order=1`, this reduces to the classical Shannon-based CAEKL mutual
    information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the Rényi CAEKL mutual information is
        calculated.
    order : float >= 0
        The order of the Rényi entropy.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the Rényi CAEKL mutual information. If
        None, then it is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        Conditioning variables are not currently supported for the Rényi CAEKL
        mutual information. Must be None or empty.

    Returns
    -------
    J_a : float
        The Rényi CAEKL mutual information.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution, if `rvs` contains
        non-existent random variables, or if `crvs` is specified.
    ValueError
        Raised if fewer than two random-variable groups are given, or if
        `order` is not a non-negative float.

    Notes
    -----
    Because each partition's value is a Rényi total correlation, only joint
    Rényi entropies are needed; no conditional Rényi entropy is involved.

    Unlike :func:`~dit.multivariate.caekl_mutual_information`, which uses the
    agglomerative principal-sequence-of-partitions algorithm, this minimizes by
    enumerating partitions. The Rényi entropy is not submodular in general, so
    the faster algorithm does not apply. The number of partitions grows as the
    Bell numbers, so this becomes expensive beyond roughly eight variables.

    References
    ----------
    Chan, C., AlBashabsheh, A., Ebrahimi, J. B., Kaced, T., & Liu, T. (2015).
    Multivariate mutual information inspired by secret-key agreement.
    Proceedings of the IEEE, 103(10), 1883-1913.

    The Rényi form substitutes H_alpha into the Shannon definition, following
    the general prescription for Rényi generalizations of information measures
    that are linear combinations of entropies: Berta, M., Seshadreesan, K. P.,
    & Wilde, M. M. (2015). Rényi generalizations of quantum information
    measures. Physical Review A, 91(2), 022333. arXiv:1502.07977.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        msg = "Conditional Rényi CAEKL mutual information is not currently supported."
        raise ditException(msg)

    rvs = list(rvs)
    if len(rvs) < 2:
        msg = "CAEKL requires at least two random-variable groups"
        raise ValueError(msg)

    def I_P(part):
        groups = [sorted(set().union(*block)) for block in part]
        return renyi_total_correlation(dist, order, rvs=groups) / (len(part) - 1)

    return min(I_P(p) for p in partitions(map(tuple, rvs)) if len(p) > 1)
