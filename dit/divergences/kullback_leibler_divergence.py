"""
The Kullback-Leibler divergence.
"""

from ..helpers import as_linear
from ..multivariate.entropy import entropy
from ..utils import unitful
from .cross_entropy import cross_entropy

__all__ = (
    "jeffreys_divergence",
    "kullback_leibler_divergence",
    "relative_entropy",
)


@unitful
def kullback_leibler_divergence(dist1, dist2, rvs=None, crvs=None):
    """
    The Kullback-Liebler divergence between `dist1` and `dist2`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the Kullback-Leibler divergence.
    dist2 : Distribution
        The second distribution in the Kullback-Leibler divergence.
    rvs : list, None
        The indexes of the random variable used to calculate the
        Kullback-Leibler divergence between. If None, then the Kullback-Leibler
        divergence is calculated over all random variables.

    Returns
    -------
    dkl : float
        The Kullback-Leibler divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.
    """
    dist1, dist2 = as_linear(dist1), as_linear(dist2)
    xh = cross_entropy(dist1, dist2, rvs, crvs)
    h = entropy(dist1, rvs, crvs)
    dkl = xh - h
    return dkl


relative_entropy = kullback_leibler_divergence


@unitful
def jeffreys_divergence(dist1, dist2, rvs=None, crvs=None):
    """
    The Jeffreys divergence between `dist1` and `dist2`.

    The Jeffreys divergence is the symmetrized Kullback-Leibler divergence:

        J(P, Q) = D_KL(P || Q) + D_KL(Q || P)

    Unlike the Kullback-Leibler divergence it is symmetric in its arguments,
    though it is still not a metric: it does not satisfy the triangle
    inequality. It is infinite unless `dist1` and `dist2` have the same
    support, since each direction requires absolute continuity.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution in the Jeffreys divergence.
    dist2 : Distribution
        The second distribution in the Jeffreys divergence.
    rvs : list, None
        The indexes of the random variable used to calculate the Jeffreys
        divergence between. If None, then the Jeffreys divergence is calculated
        over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    dj : float
        The Jeffreys divergence between `dist1` and `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.

    References
    ----------
    Jeffreys, H. (1946). An invariant form for the prior probability in
    estimation problems. Proceedings of the Royal Society of London A,
    186(1007), 453-461.
    """
    forward = kullback_leibler_divergence(dist1, dist2, rvs, crvs)
    reverse = kullback_leibler_divergence(dist2, dist1, rvs, crvs)
    dj = forward + reverse
    return dj
