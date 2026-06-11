"""
Trivial endpoint channels: the noiseless and zero-capacity extremes.
"""

from ..exceptions import ditException
from ._util import conditional_from_matrix

__all__ = (
    "identity_channel",
    "useless_channel",
)


def identity_channel(n=2):
    """
    The noiseless channel over an ``n``-symbol alphabet.

    Every symbol is received unchanged, so the channel has capacity
    :math:`\\log_2 n`.

    Parameters
    ----------
    n : int
        The alphabet size (``n >= 1``).

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, ..., n - 1}``.
    """
    if n < 1:
        raise ditException("The alphabet size n must be at least 1.")
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return conditional_from_matrix(P, list(range(n)), list(range(n)))


def useless_channel(n=2):
    """
    The zero-capacity channel over an ``n``-symbol alphabet.

    The output is uniform and independent of the input, so the channel carries no
    information and has capacity ``0``.

    Parameters
    ----------
    n : int
        The alphabet size (``n >= 1``).

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, ..., n - 1}``.
    """
    if n < 1:
        raise ditException("The alphabet size n must be at least 1.")
    P = [[1.0 / n for _ in range(n)] for _ in range(n)]
    return conditional_from_matrix(P, list(range(n)), list(range(n)))
