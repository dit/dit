"""
The cross mutual information and its multivariate generalizations.

The cross mutual information [Gohil, Cliff, Shine, Fulcher, and Lizier, "Cross
Mutual Information", IEEE Information Theory Workshop 2025, arXiv:2507.15372]
measures the expected strength of the dependence between random variables
exhibited by samples from a *test* distribution ``p``, where the pointwise
information of each sample is evaluated using a *reference* distribution ``q``:

.. math::

   CI_{pq} = \\mathbb{E}_{p}\\{i_q(x;y)\\}
           = \\sum_{x,y} p(x,y) \\log_2 \\frac{q(x,y)}{q(x)q(y)}

When ``p == q`` the cross mutual information reduces to the ordinary mutual
information. Each of the standard multivariate mutual informations
(co-information, total correlation, dual total correlation, and CAEKL mutual
information) is a signed sum of joint entropies; the corresponding cross measure
replaces each entropy with the analogous cross entropy between ``p`` and ``q``.
Unlike their conventional counterparts, the cross measures can be negative.
"""

from ..divergences.cross_entropy import cross_entropy
from ..helpers import normalize_rvs
from ..utils import partitions, powerset

__all__ = (
    "cross_caekl_mutual_information",
    "cross_coinformation",
    "cross_dual_total_correlation",
    "cross_total_correlation",
)


def cross_coinformation(dist, ref_dist, rvs=None, crvs=None):
    """
    Calculates the cross co-information, the cross-distribution generalization
    of the (multivariate) mutual information.

    For two random variables this is the cross mutual information of [Gohil et
    al., arXiv:2507.15372].

    Parameters
    ----------
    dist : Distribution
        The test distribution `p`, over which the expectation is taken.
    ref_dist : Distribution
        The reference distribution `q`, used to evaluate the pointwise
        information of each outcome.
    rvs : list, None
        The indexes of the random variable used to calculate the cross
        co-information between. If None, then the cross co-information is
        calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.

    Returns
    -------
    CI : float
        The cross co-information.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    def cross_entropy_of(rvs):
        """Cross entropy of the union of `rvs`, conditioned on `crvs`."""
        return cross_entropy(dist, ref_dist, list(set().union(*rvs)), crvs)

    CI = sum((-1) ** (len(Xs) + 1) * cross_entropy_of(Xs) for Xs in powerset(rvs))

    return CI


def cross_total_correlation(dist, ref_dist, rvs=None, crvs=None):
    """
    Computes the cross total correlation, the cross-distribution generalization
    of the total correlation.

    Parameters
    ----------
    dist : Distribution
        The test distribution `p`, over which the expectation is taken.
    ref_dist : Distribution
        The reference distribution `q`, used to evaluate the pointwise
        information of each outcome.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the cross total correlation. If None, then
        the cross total correlation is calculated over all random variables,
        which is equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    CT : float
        The cross total correlation.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    one = sum(cross_entropy(dist, ref_dist, rv, crvs) for rv in rvs)
    two = cross_entropy(dist, ref_dist, list(set().union(*rvs)), crvs)
    CT = one - two

    return CT


def cross_dual_total_correlation(dist, ref_dist, rvs=None, crvs=None):
    """
    Calculates the cross dual total correlation, the cross-distribution
    generalization of the dual total correlation.

    Parameters
    ----------
    dist : Distribution
        The test distribution `p`, over which the expectation is taken.
    ref_dist : Distribution
        The reference distribution `q`, used to evaluate the pointwise
        information of each outcome.
    rvs : list, None
        The indexes of the random variable used to calculate the cross dual
        total correlation. If None, then the cross dual total correlation is
        calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are condition on.

    Returns
    -------
    CB : float
        The cross dual total correlation.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    others = lambda rv, rvs: set(set().union(*rvs)) - set(rv)

    one = cross_entropy(dist, ref_dist, list(set().union(*rvs)), crvs)
    two = sum(cross_entropy(dist, ref_dist, rv, list(others(rv, rvs).union(crvs))) for rv in rvs)
    CB = one - two

    return CB


def cross_caekl_mutual_information(dist, ref_dist, rvs=None, crvs=None):
    """
    Calculates the cross CAEKL mutual information, the cross-distribution
    generalization of the CAEKL mutual information.

    Parameters
    ----------
    dist : Distribution
        The test distribution `p`, over which the expectation is taken.
    ref_dist : Distribution
        The reference distribution `q`, used to evaluate the pointwise
        information of each outcome.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the cross CAEKL mutual information. If None,
        then the cross CAEKL mutual information is calculated over all random
        variables, which is equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    CJ : float
        The cross CAEKL mutual information.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    joint = cross_entropy(dist, ref_dist, list(set().union(*rvs)), crvs)

    def CI_P(part):
        a = sum(cross_entropy(dist, ref_dist, list(p), crvs) for p in part)
        return (a - joint) / (len(part) - 1)

    candidates = [CI_P(p) for p in partitions(map(tuple, rvs)) if len(p) > 1]

    return min(candidates)
