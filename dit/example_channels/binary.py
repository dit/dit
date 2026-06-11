"""
Binary-input example channels.
"""

from ..exceptions import ditException
from ._util import conditional_from_matrix

__all__ = (
    "binary_asymmetric_channel",
    "binary_erasure_channel",
    "binary_symmetric_channel",
    "binary_symmetric_erasure_channel",
    "z_channel",
)

# The erasure symbol; the integer just past the binary input alphabet {0, 1}.
ERASURE = 2


def binary_symmetric_channel(p):
    """
    The binary symmetric channel with crossover probability ``p``.

    Each transmitted bit is independently flipped with probability ``p``. Its
    capacity is :math:`1 - H_b(p)`.

    Parameters
    ----------
    p : float
        The probability that a transmitted bit is flipped.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, 1}``.
    """
    if not 0 <= p <= 1:
        raise ditException("The crossover probability p must lie in [0, 1].")
    P = [[1 - p, p], [p, 1 - p]]
    return conditional_from_matrix(P, [0, 1], [0, 1])


def binary_erasure_channel(epsilon):
    """
    The binary erasure channel with erasure probability ``epsilon``.

    Each transmitted bit is independently erased with probability ``epsilon`` and
    otherwise received unchanged. Its capacity is :math:`1 - \\epsilon`.

    Parameters
    ----------
    epsilon : float
        The probability that a transmitted bit is erased.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` with output alphabet
        ``{0, 1, 2}``, where ``2`` denotes an erasure.
    """
    if not 0 <= epsilon <= 1:
        raise ditException("The erasure probability epsilon must lie in [0, 1].")
    P = [[1 - epsilon, 0, epsilon], [0, 1 - epsilon, epsilon]]
    return conditional_from_matrix(P, [0, 1], [0, 1, ERASURE])


def z_channel(p):
    """
    The Z-channel with crossover probability ``p``.

    A ``0`` is always received correctly; a ``1`` is flipped to ``0`` with
    probability ``p``. This is the canonical binary asymmetric channel.

    Parameters
    ----------
    p : float
        The probability that a transmitted ``1`` is received as ``0``.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, 1}``.
    """
    if not 0 <= p <= 1:
        raise ditException("The crossover probability p must lie in [0, 1].")
    P = [[1, 0], [p, 1 - p]]
    return conditional_from_matrix(P, [0, 1], [0, 1])


def binary_asymmetric_channel(p0, p1):
    """
    The binary asymmetric channel.

    A ``0`` is flipped to ``1`` with probability ``p0``; a ``1`` is flipped to
    ``0`` with probability ``p1``. The binary symmetric channel is the case
    ``p0 == p1`` and the Z-channel is the case ``p0 == 0``.

    Parameters
    ----------
    p0 : float
        The probability that a transmitted ``0`` is received as ``1``.
    p1 : float
        The probability that a transmitted ``1`` is received as ``0``.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, 1}``.
    """
    if not 0 <= p0 <= 1 or not 0 <= p1 <= 1:
        raise ditException("The crossover probabilities must lie in [0, 1].")
    P = [[1 - p0, p0], [p1, 1 - p1]]
    return conditional_from_matrix(P, [0, 1], [0, 1])


def binary_symmetric_erasure_channel(p, epsilon):
    """
    The binary symmetric error-and-erasure channel.

    A transmitted bit is erased with probability ``epsilon`` and flipped with
    probability ``p``; the two events are disjoint, so ``p + epsilon <= 1``.

    Parameters
    ----------
    p : float
        The probability that a transmitted bit is flipped.
    epsilon : float
        The probability that a transmitted bit is erased.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` with output alphabet
        ``{0, 1, 2}``, where ``2`` denotes an erasure.
    """
    if p < 0 or epsilon < 0 or p + epsilon > 1:
        raise ditException("Require p >= 0, epsilon >= 0, and p + epsilon <= 1.")
    P = [
        [1 - p - epsilon, p, epsilon],
        [p, 1 - p - epsilon, epsilon],
    ]
    return conditional_from_matrix(P, [0, 1], [0, 1, ERASURE])
