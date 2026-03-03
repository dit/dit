"""
The CAEKL mutual information, as define [Chan, Chung, et al. "Multivariate
Mutual Information Inspired by Secret-Key Agreement." Proceedings of the IEEE
103.10 (2015): 1883-1913].
"""

from ..helpers import normalize_rvs
from ..utils import partitions, unitful
from .entropy import entropy

__all__ = ("caekl_mutual_information",)


@unitful
def caekl_mutual_information(dist, rvs=None, crvs=None):
    """
    Calculates the Chan-AlBashabsheh-Ebrahimi-Kaced-Liu mutual information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the CAEKL mutual information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    J : float
        The CAEKL mutual information.

    Examples
    --------
    >>> d = dit.example_dists.Xor()
    >>> dit.multivariate.caekl_mutual_information(d)
    0.5
    >>> dit.multivariate.caekl_mutual_information(d, rvs=[[0], [1]])
    0.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existant random variables.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    H = entropy(dist, rvs, crvs)

    def I_P(part):
        a = sum(entropy(dist, rvs=p, crvs=crvs) for p in part)
        return (a - H) / (len(part) - 1)

    J = min(I_P(p) for p in partitions(map(tuple, rvs)) if len(p) > 1)

    return J
