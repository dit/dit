"""
Constructors for classical linear block codes.
"""

import itertools
from math import comb

import numpy as np

from ..exceptions import ditException
from . import _gf2
from .linear import LinearCode

__all__ = (
    "golay",
    "hamming",
    "parity_check",
    "reed_muller",
    "repetition",
)


def repetition(n, channel=None):
    """
    The ``[n, 1, n]`` repetition code.

    Parameters
    ----------
    n : int
        The block length.
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : LinearCode
    """
    if n < 1:
        raise ditException("The repetition length must be at least 1.")
    G = np.ones((1, n), dtype=int)
    return LinearCode(G, channel=channel)


def parity_check(k, channel=None):
    """
    The ``[k+1, k, 2]`` single-parity-check code.

    Parameters
    ----------
    k : int
        The number of information bits.
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : LinearCode
    """
    if k < 1:
        raise ditException("The parity-check dimension must be at least 1.")
    G = np.hstack([np.eye(k, dtype=int), np.ones((k, 1), dtype=int)])
    return LinearCode(G, channel=channel)


def hamming(r, channel=None):
    """
    The ``[2^r - 1, 2^r - 1 - r, 3]`` Hamming code.

    Parameters
    ----------
    r : int
        The number of parity bits (``r >= 2``).
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : LinearCode
    """
    if r < 2:
        raise ditException("The Hamming parameter r must be at least 2.")
    columns = [[(value >> bit) & 1 for bit in range(r)] for value in range(1, 2**r)]
    H = np.array(columns, dtype=int).T
    G = _gf2.nullspace(H)
    return LinearCode(G, channel=channel)


def reed_muller(r, m, channel=None):
    """
    The Reed-Muller code ``RM(r, m)``.

    Parameters
    ----------
    r : int
        The order (``0 <= r <= m``).
    m : int
        The number of variables; the block length is ``2^m``.
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : LinearCode
    """
    if not 0 <= r <= m:
        raise ditException("Reed-Muller requires 0 <= r <= m.")
    n = 2**m
    points = list(itertools.product((0, 1), repeat=m))
    rows = []
    for degree in range(r + 1):
        for subset in itertools.combinations(range(m), degree):
            row = [int(all(point[i] for i in subset)) for point in points]
            rows.append(row)
    G = np.array(rows, dtype=int)
    assert G.shape == (sum(comb(m, i) for i in range(r + 1)), n)
    return LinearCode(G, channel=channel)


# Generator polynomial of the [23, 12, 7] binary Golay code:
# g(x) = x^11 + x^9 + x^7 + x^6 + x^5 + x + 1.
_GOLAY_G = [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]


def golay(extended=False, channel=None):
    """
    The binary Golay code.

    Parameters
    ----------
    extended : bool
        If True, return the extended ``[24, 12, 8]`` code; otherwise the perfect
        ``[23, 12, 7]`` code.
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : LinearCode
    """
    k, n = 12, 23
    G = np.zeros((k, n), dtype=int)
    for i in range(k):
        G[i, i : i + len(_GOLAY_G)] = _GOLAY_G
    if extended:
        parity = G.sum(axis=1) % 2
        G = np.hstack([G, parity[:, None]])
    return LinearCode(G, channel=channel)
