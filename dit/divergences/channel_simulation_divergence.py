"""
The channel simulation divergence and its Rényi generalization.

For :math:`P \\ll Q` the width function :math:`w(\\ell) = Q(dP/dQ \\geq \\ell)`,
:math:`\\ell \\geq 0`, is a probability density on :math:`[0, \\infty)`. The
channel simulation divergence :cite:`goc2024channel` is its differential
entropy, and its Rényi generalization :cite:`flamich2026functional` is its
differential Rényi entropy.
"""

import numpy as np

from ..utils import unitful
from .generalized_divergences import _union_pmfs

__all__ = (
    "channel_simulation_divergence",
    "channel_simulation_divergence_pmf",
)


def _validate_order(order):
    """The Rényi forms are defined for orders in (0, 1); order 1 is the Shannon form."""
    if not 0 < order <= 1:
        msg = "the order must lie in (0, 1]"
        raise ValueError(msg)


def _width_steps(ratios, q):
    """
    Decompose the width function of each column of ``ratios`` into steps.

    Parameters
    ----------
    ratios : np.ndarray, shape (n, m)
        Column ``j`` is (proportional to) a likelihood ratio :math:`dP_j/dQ` on
        the ``n`` atoms of :math:`Q`.
    q : np.ndarray, shape (n,)
        The positive masses of :math:`Q`.

    Returns
    -------
    levels : np.ndarray, shape (n, m)
        With each column sorted as :math:`r_{(0)} \\geq \\cdots \\geq r_{(n-1)}`,
        ``levels[i]`` is the :math:`Q`-mass of the ``i + 1`` largest ratios.
    lengths : np.ndarray, shape (n, m)
        ``lengths[i]`` is :math:`r_{(i)} - r_{(i+1)}`, with :math:`r_{(n)} = 0`.
        The width function equals ``levels[i]`` on an interval of this length.
    """
    order = np.argsort(-ratios, axis=0, kind="stable")
    ratios = np.take_along_axis(ratios, order, axis=0)
    levels = np.cumsum(q[order], axis=0)
    lengths = ratios - np.vstack([ratios[1:], np.zeros((1, ratios.shape[1]))])
    return levels, lengths


def _width_integrals(levels, lengths, order):
    """
    Per column: the differential entropy (bits) of the width function at order
    1, and the integral of its ``order``-th power otherwise.
    """
    if order == 1:
        return -(lengths * levels * np.log2(levels)).sum(axis=0)
    return (lengths * levels**order).sum(axis=0)


def channel_simulation_divergence_pmf(p, q, alpha=1):
    """
    The channel simulation divergence of ``p`` from ``q``.

    Parameters
    ----------
    p : array_like
        The first pmf.
    q : array_like
        The second pmf, over the same outcomes as ``p``.
    alpha : float
        The order, in (0, 1]. ``alpha=1`` (the default) is the channel
        simulation divergence; smaller orders give its Rényi generalization.

    Returns
    -------
    dcs : float
        The divergence in bits; infinite unless ``p`` is absolutely continuous
        with respect to ``q``.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1].
    """
    _validate_order(alpha)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    support = q > 0
    if np.any(p[~support] > 0):
        return np.inf
    levels, lengths = _width_steps((p[support] / q[support])[:, np.newaxis], q[support])
    value = float(_width_integrals(levels, lengths, alpha)[0])
    if alpha == 1:
        return value
    return float(np.log2(value) / (1 - alpha))


@unitful
def channel_simulation_divergence(dist1, dist2, alpha=1, rvs=None):
    """
    The channel simulation divergence of `dist1` from `dist2`.

    For :math:`P \\ll Q` with likelihood ratio :math:`r = dP/dQ`, the width
    function :math:`w(\\ell) = Q(r \\geq \\ell)` integrates to one over
    :math:`\\ell \\geq 0`. The channel simulation divergence is its
    differential entropy :cite:`goc2024channel` (Definition 2 of
    :cite:`flamich2026functional`)

    .. math::

        D_{CS}(P \\| Q) = -\\int_0^\\infty w(\\ell) \\log_2 w(\\ell)\\, d\\ell,

    and for :math:`\\alpha \\in (0, 1)` the Rényi channel simulation divergence
    (Definition 9 of :cite:`flamich2026functional`) is its differential Rényi
    entropy

    .. math::

        D^{CS}_\\alpha(P \\| Q) = \\frac{1}{1 - \\alpha}
            \\log_2 \\int_0^\\infty w(\\ell)^\\alpha\\, d\\ell.

    It sandwiches the Kullback-Leibler divergence (Proposition 1 of
    :cite:`flamich2026functional`):

    .. math::

        D(P \\| Q) \\leq D_{CS}(P \\| Q) \\leq D(P \\| Q) + \\log_2(D(P \\| Q) + 1) + 1.

    Averaged over :math:`P_{X|Y=y}` against :math:`P_X`, it is the functional
    information :func:`dit.other.functional_information`.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution, :math:`P`.
    dist2 : Distribution
        The second distribution, :math:`Q`.
    alpha : float
        The order, in (0, 1]. ``alpha=1`` (the default) is the channel
        simulation divergence.
    rvs : list, None
        The indexes of the random variables used to calculate the divergence.
        If None, then it is calculated over all random variables.

    Returns
    -------
    dcs : float
        The channel simulation divergence; infinite unless `dist1` is
        absolutely continuous with respect to `dist2`.

    Raises
    ------
    ditException
        Raised if either `dist1` or `dist2` doesn't have `rvs` or, if `rvs` is
        None, if `dist2` has an outcome length different than `dist1`.
    ValueError
        If ``alpha`` is not in (0, 1].
    """
    p, q = _union_pmfs(dist1, dist2, rvs)
    return channel_simulation_divergence_pmf(p, q, alpha)
