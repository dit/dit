"""
The copy mutual information, as defined by Kolchinsky & Corominas-Murtra.
"""

from .pmf import relative_entropy
from ..utils import unitful


__all__ = [
    'copy_mutual_information',
]


def binary_kullback_leibler_divergence(p, q):
    """
    Compute the binary Killback Leibler divergence.

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
    return relative_entropy([p, 1-p], [q, 1-q])


@unitful
def specific_copy_mutual_information(p_Y_g_x, p_Y, x):
    """
    Compute the specific copy mutual information. Roughly it is the
    portion of the specific mututal information which results from X = Y = x.

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


def copy_mutual_information(dist, X, Y, rv_mode=None):
    """
    Computes the copy mutual information. Roughly, it is the
    portion of the mutual information which results from X = Y.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    X : iterable
        The indicies to consider as X.
    Y : iterable
        The indicies to consider as Y.
    rv_mode : str, None
        Specifies how to interpret ``crvs`` and ``rvs``. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements
        of ``crvs`` and ``rvs`` are interpreted as random variable indices.
        If equal to 'names', the the elements are interpreted as random
        varible names. If ``None``, then the value of ``self._rv_mode`` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    Icopy : float
        The copy mutual information of x.
    """
    p_Y = dist.marginal(Y, rv_mode=rv_mode)
    marg, cdists = dist.condition_on(X, rvs=Y, rv_mode=rv_mode)
    return sum([marg[x]*specific_copy_mutual_information(cdist, p_Y, x) for x, cdist in zip(marg.outcomes, cdists)])
