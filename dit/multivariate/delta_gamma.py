"""
The Delta^k and Gamma^k measures, as defined by Varley.

These are a parameterized family of higher-order information measures that
unify the S-information, dual total correlation, and (negative) O-information
(Delta^k), along with their entropic conjugates (Gamma^k).
"""

from .dual_total_correlation import dual_total_correlation
from .total_correlation import total_correlation

__all__ = (
    "delta_k",
    "gamma_k",
)


def delta_k(dist, k, rvs=None, crvs=None):
    """
    Compute the Delta^k measure, a parameterized family of higher-order
    information measures.

    It is defined as :math:`\\Delta^k = \\mathcal{S} - k\\mathcal{T}`, where
    :math:`\\mathcal{S}` is the S-information and :math:`\\mathcal{T}` is the
    total correlation. Since the S-information is the sum of the total
    correlation and the dual total correlation :math:`\\mathcal{D}`, this is
    equivalent to :math:`\\Delta^k = \\mathcal{D} + (1 - k)\\mathcal{T}`.

    Special cases recover known measures: :math:`\\Delta^0` is the
    S-information, :math:`\\Delta^1` is the dual total correlation, and
    :math:`\\Delta^2` is the negative O-information. For larger `k`, the
    measure is sensitive to increasingly high-order synergies.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the Delta^k measure is calculated.
    k : int
        The order parameter.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the Delta^k measure. If None, then it is
        calculated over all random variables, which is equivalent to passing
        `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    D_k : float
        The Delta^k measure.

    Examples
    --------
    >>> d = dit.example_dists.n_mod_m(5, 2)
    >>> dit.multivariate.delta_k(d, 2)
    3.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    t = total_correlation(dist=dist, rvs=rvs, crvs=crvs)
    d = dual_total_correlation(dist=dist, rvs=rvs, crvs=crvs)
    return d + (1 - k) * t


def gamma_k(dist, k, rvs=None, crvs=None):
    """
    Compute the Gamma^k measure, the entropic conjugate of the Delta^k measure.

    It is defined as :math:`\\Gamma^k = \\mathcal{S} - k\\mathcal{D}`, where
    :math:`\\mathcal{S}` is the S-information and :math:`\\mathcal{D}` is the
    dual total correlation. Since the S-information is the sum of the total
    correlation :math:`\\mathcal{T}` and the dual total correlation, this is
    equivalent to :math:`\\Gamma^k = \\mathcal{T} + (1 - k)\\mathcal{D}`.

    Special cases recover known measures: :math:`\\Gamma^0` is the
    S-information, :math:`\\Gamma^1` is the total correlation, and
    :math:`\\Gamma^2` is the O-information. For larger `k`, the measure is
    sensitive to increasingly high-order redundancies.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the Gamma^k measure is calculated.
    k : int
        The order parameter.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the Gamma^k measure. If None, then it is
        calculated over all random variables, which is equivalent to passing
        `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    G_k : float
        The Gamma^k measure.

    Examples
    --------
    >>> d = dit.example_dists.n_mod_m(5, 2)
    >>> dit.multivariate.gamma_k(d, 2)
    -3.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    t = total_correlation(dist=dist, rvs=rvs, crvs=crvs)
    d = dual_total_correlation(dist=dist, rvs=rvs, crvs=crvs)
    return t + (1 - k) * d
