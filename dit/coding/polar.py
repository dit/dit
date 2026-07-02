"""
Polar codes with successive-cancellation decoding (Arikan, 2009).

A :class:`PolarCode` freezes the least reliable synthesized bit-channels (ranked
by their Bhattacharyya parameter) and carries information on the rest. Encoding is
the Arikan polar transform; decoding is successive cancellation using channel
log-likelihood ratios.
"""

from math import log2

import numpy as np

from ..exceptions import ditException
from ._util import polar_transform as _polar_transform
from .linear import LinearCode

__all__ = (
    "PolarCode",
    "polar",
)


def _bhattacharyya_zs(z0, m):
    """The Bhattacharyya parameters of the ``2^m`` synthesized channels."""
    z = [z0]
    for _ in range(m):
        z = [value for zi in z for value in (2 * zi - zi * zi, zi * zi)]
    return z


class PolarCode(LinearCode):
    """
    A polar code of length ``n = 2^m`` with ``k`` information bits.

    Parameters
    ----------
    n : int
        The block length; must be a power of two.
    k : int
        The number of information bits.
    channel : Distribution
        The channel used both to select the frozen set and to decode.
    """

    def __init__(self, n, k, channel):
        if n & (n - 1) != 0:
            raise ditException("The polar block length n must be a power of two.")
        if not 0 < k <= n:
            raise ditException("The polar dimension k must satisfy 0 < k <= n.")
        m = int(log2(n))

        from ._channel import bhattacharyya

        z = _bhattacharyya_zs(bhattacharyya(channel), m)
        # The most reliable channels (smallest Bhattacharyya) carry information.
        order = sorted(range(n), key=lambda i: z[i])
        self.info_indices = sorted(order[:k])
        self.frozen = np.ones(n, dtype=bool)
        self.frozen[self.info_indices] = False

        rows = []
        for i in self.info_indices:
            e = [0] * n
            e[i] = 1
            rows.append(_polar_transform(e))
        G = np.array(rows, dtype=int)
        super().__init__(G, channel=channel)

    @property
    def message_length(self):
        return len(self.info_indices)

    def decode(self, received, channel=None):
        """
        Decode by successive cancellation using channel LLRs.
        """
        channel = channel if channel is not None else self.channel
        if channel is None:
            raise ditException("Polar decoding requires a channel.")
        from ._channel import log_likelihoods

        llr_map = log_likelihoods(channel)
        llrs = [llr_map[y] for y in received]
        decisions = [None] * self.n
        self._sc(llrs, list(range(self.n)), decisions)
        return [decisions[i] for i in self.info_indices]

    def _sc(self, llrs, indices, decisions):
        """Recursive successive cancellation; returns this subtree's codeword bits."""
        if len(indices) == 1:
            i = indices[0]
            if self.frozen[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [_f(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = self._sc(left_llr, indices[:half], decisions)
        right_llr = [_g(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)]
        enc_right = self._sc(right_llr, indices[half:], decisions)
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right


def _f(a, b):
    """The check-node (min-sum) update for successive cancellation."""
    return float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))


def _g(a, b, u):
    """The variable-node update for successive cancellation."""
    return b - a if u else b + a


def polar(n, k, channel):
    """
    Build a polar code.

    Parameters
    ----------
    n : int
        The block length; must be a power of two.
    k : int
        The number of information bits.
    channel : Distribution
        The channel used to select the frozen set and to decode.

    Returns
    -------
    code : PolarCode
    """
    return PolarCode(n, k, channel)
