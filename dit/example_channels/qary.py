"""
q-ary example channels.
"""

from ..exceptions import ditException
from ._util import conditional_from_matrix

__all__ = (
    "noisy_typewriter",
    "q_ary_erasure_channel",
    "q_ary_symmetric_channel",
)


def q_ary_symmetric_channel(q, p):
    """
    The q-ary symmetric channel.

    A symbol is received correctly with probability ``1 - p`` and is otherwise
    received as one of the other ``q - 1`` symbols, chosen uniformly. Its capacity
    is :math:`\\log_2 q - H_b(p) - p \\log_2(q - 1)`. The binary symmetric channel
    is the case ``q == 2``.

    Parameters
    ----------
    q : int
        The alphabet size (``q >= 2``).
    p : float
        The total probability that a symbol is received incorrectly.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, ..., q - 1}``.
    """
    if q < 2:
        raise ditException("The alphabet size q must be at least 2.")
    if not 0 <= p <= 1:
        raise ditException("The error probability p must lie in [0, 1].")
    off = p / (q - 1)
    P = [[1 - p if i == j else off for j in range(q)] for i in range(q)]
    return conditional_from_matrix(P, list(range(q)), list(range(q)))


def q_ary_erasure_channel(q, epsilon):
    """
    The q-ary erasure channel.

    A symbol is erased with probability ``epsilon`` and otherwise received
    unchanged. Its capacity is :math:`(1 - \\epsilon) \\log_2 q`. The binary
    erasure channel is the case ``q == 2``.

    Parameters
    ----------
    q : int
        The alphabet size (``q >= 2``).
    epsilon : float
        The probability that a symbol is erased.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` with output alphabet
        ``{0, ..., q - 1, q}``, where ``q`` denotes an erasure.
    """
    if q < 2:
        raise ditException("The alphabet size q must be at least 2.")
    if not 0 <= epsilon <= 1:
        raise ditException("The erasure probability epsilon must lie in [0, 1].")
    erasure = q
    P = [[(1 - epsilon if j == i else 0) for j in range(q)] + [epsilon] for i in range(q)]
    return conditional_from_matrix(P, list(range(q)), list(range(q)) + [erasure])


def noisy_typewriter(n=26):
    """
    The noisy typewriter channel (Cover & Thomas).

    Each of the ``n`` letters is received either unchanged or as the next letter
    (cyclically), each with probability one half. Its capacity is
    :math:`\\log_2(n / 2)`, achieved by using every other input letter.

    Parameters
    ----------
    n : int
        The alphabet size (``n >= 2``).

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, ..., n - 1}``.
    """
    if n < 2:
        raise ditException("The alphabet size n must be at least 2.")
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P[i][i] = 0.5
        P[i][(i + 1) % n] = 0.5
    return conditional_from_matrix(P, list(range(n)), list(range(n)))
