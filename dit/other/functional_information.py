"""
The functional information and its Rényi generalization.

The functional information :math:`I_F(X \\to Y)` :cite:`flamich2026functional`
characterizes one-shot channel simulation: the least expected entropy
:math:`H[Y | Z]` of a functional representation :math:`Y = f(X, Z)` with
:math:`Z` independent of :math:`X` lies between :math:`I_F(X \\to Y)` and
:math:`I_F(X \\to Y) + \\log_2 e`.
"""

import numpy as np

from ..divergences.channel_simulation_divergence import _validate_order, _width_integrals, _width_steps
from ..utils import unitful
from .sibson_mutual_information import _joint_pmf

__all__ = (
    "functional_information",
    "functional_information_pmf",
)


def functional_information_pmf(p_xy, order=1):
    """
    Compute the functional information :math:`I_F(X \\to Y)` of order ``order``.

    Parameters
    ----------
    p_xy : array_like, shape (n_x, n_y)
        Joint PMF with axis 0 indexing ``X`` and axis 1 indexing ``Y``.
    order : float
        The order, in (0, 1]. ``order=1`` (the default) is the functional
        information; smaller orders give the Rényi functional information.

    Returns
    -------
    I_F : float
        The functional information in bits.

    Raises
    ------
    ValueError
        If ``order`` is not in (0, 1].
    """
    _validate_order(order)
    p_xy = np.asarray(p_xy, dtype=float)
    p_xy = p_xy[p_xy.sum(axis=1) > 0]
    p_x = p_xy.sum(axis=1)
    # Column y of p(y|x) is p(y) times the likelihood ratio dP_{X|Y=y} / dP_X, so
    # its steps already carry the weight p(y) of the expectation over Y.
    levels, lengths = _width_steps(p_xy / p_x[:, np.newaxis], p_x)
    total = float(_width_integrals(levels, lengths, order).sum())
    if order == 1:
        return total
    return float(np.log2(total) / (1 - order))


@unitful
def functional_information(dist, rvs_X, rvs_Y, order=1):
    """
    Compute the functional information :math:`I_F(X \\to Y)` of order ``order``.

    The functional information of the channel :math:`X \\to Y`
    :cite:`flamich2026functional` (Definition 3) is the channel simulation
    divergence (:func:`dit.divergences.channel_simulation_divergence`) of
    :math:`P_{X|Y}` from :math:`P_X`, averaged over :math:`Y`:

    .. math::

        I_F(X \\to Y) = \\mathbb{E}_Y\\left[h(w_Y)\\right],
        \\qquad w_y(\\ell) = P_X\\left(\\frac{p(y|X)}{p(y)} \\geq \\ell\\right),

    where :math:`h` is differential entropy. It lies within
    :math:`\\log_2 e` bits of the least expected entropy :math:`H[Y | Z]` of
    any functional representation :math:`Y = f(X, Z)` with :math:`Z`
    independent of :math:`X` (Theorem 1). For finite alphabets that least
    entropy is the ``inf_h`` of
    :func:`dit.rate_distortion.excess_functional_information`, and the two
    coincide when either alphabet is binary.

    It is asymmetric in :math:`(X, Y)`, and it is close to the mutual
    information (Proposition 3):

    .. math::

        I[X : Y] \\leq I_F(X \\to Y) \\leq I[X : Y] + \\log_2(I[X : Y] + 1) + 1,

    with equality on the left exactly when the channel is singular, i.e. when
    :math:`p(y|x) / p(y)` depends only on :math:`y` wherever it is positive. In
    particular :math:`I_F(X \\to X) = H[X]`. Like the mutual information it
    obeys data processing, is concave in :math:`p(x)` and convex in
    :math:`p(y|x)` (Propositions 4 and 5).

    For :math:`\\alpha \\in (0, 1)` the Rényi functional information
    (Definition 10) is

    .. math::

        I_{F,\\alpha}(X \\to Y) = \\frac{1}{1 - \\alpha}
            \\log_2 \\mathbb{E}_Y\\left[\\int_0^\\infty w_Y(\\ell)^\\alpha\\, d\\ell\\right],

    which tends to :math:`I_F` as :math:`\\alpha \\to 1`, gives
    :math:`I_{F,\\alpha}(X \\to X) = H_\\alpha[X]` (Proposition 10), and
    bounds the Sibson mutual information of order :math:`1/\\alpha` from above,
    with equality exactly for singular channels (Proposition 11).

    Sorting each column of :math:`p(y|x)` as
    :math:`k_{(0)} \\geq \\cdots \\geq k_{(n-1)} \\geq k_{(n)} = 0`, with
    :math:`S_i` the :math:`P_X`-mass of the first :math:`i + 1`, gives the closed
    form (Lemma 3)

    .. math::

        I_F(X \\to Y) = \\sum_y \\sum_i \\left(k_{(i)} - k_{(i+1)}\\right) \\eta(S_i),
        \\qquad \\eta(s) = -s \\log_2 s.

    Parameters
    ----------
    dist : Distribution
        The joint distribution.
    rvs_X : list
        Indexes of the random variables defining the channel input ``X``.
    rvs_Y : list
        Indexes of the random variables defining the channel output ``Y``.
    order : float
        The order, in (0, 1]. ``order=1`` (the default) is the functional
        information.

    Returns
    -------
    I_F : float
        The functional information.

    Raises
    ------
    ValueError
        If ``order`` is not in (0, 1].
    """
    _validate_order(order)
    return functional_information_pmf(_joint_pmf(dist, rvs_X, rvs_Y), order)
