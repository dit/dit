"""
The copy mutual information, as defined by Kolchinsky & Corominas-Murtra.
"""

from ..utils import unitful
from .pmf import relative_entropy

__all__ = ("copy_mutual_information",)


def binary_kullback_leibler_divergence(p, q):
    """
    Compute the binary Killback-Leibler divergence.

    Parameters
    ----------
    p : float
        The first probability.
    q : float
        The second probability.

    Returns
    -------
    dkl : float
        The binary Kullback-Leibler divergence.
    """
    return relative_entropy([p, 1 - p], [q, 1 - q])


@unitful
def specific_copy_mutual_information(p_Y_g_x, p_Y, x):
    """
    Compute the specific copy mutual information. Roughly it is the
    portion of the specific mutual information which results from X = Y = x.

    Parameters
    ----------
    p_Y_g_x : Distribution
        The probability p(Y|X=x).
    p_Y : Distribution
        The probability p(Y).
    x : event
        An event in the sample space of X, Y.

    Returns
    -------
    Icopy : float
        The specific copy mutual information of x.
    """
    py = p_Y[x]
    pygx = p_Y_g_x[x]
    if pygx > py:
        return binary_kullback_leibler_divergence(pygx, py)
    else:
        return 0


def copy_mutual_information(dist, X, Y):
    """
    Computes the copy mutual information. Roughly, it is the
    portion of the mutual information which results from :math:`X = Y`.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    X : iterable
        The indicies to consider as X.
    Y : iterable
        The indicies to consider as Y.

    Returns
    -------
    Icopy : float
        The copy mutual information of x.
    """
    p_Y = dist.marginal(Y)
    marg, cdists = dist.condition_on(X, rvs=Y)
    return sum(
        marg[x] * specific_copy_mutual_information(cdist, p_Y, x)
        for x, cdist in zip(marg.outcomes, cdists, strict=True)
    )
