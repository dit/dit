"""
Linear block codes over GF(2).

A :class:`LinearCode` is defined by a generator matrix ``G`` (``k x n``); the
parity-check matrix ``H`` is derived as a basis for the null space of ``G``. This
is the workhorse for the classical binary block codes (repetition, parity-check,
Hamming, Reed-Muller, Golay) and the base for LDPC and polar codes.
"""

import itertools
from collections import Counter

import numpy as np

from ..exceptions import ditException
from . import _gf2
from .base import ChannelCoding

__all__ = ("LinearCode",)

_MAX_ENUMERATION = 20


class LinearCode(ChannelCoding):
    """
    A linear block code over GF(2).

    Parameters
    ----------
    G : array-like
        A ``k x n`` binary generator matrix of full row rank.
    channel : Distribution, None
        A default channel, as a conditional distribution ``p(Y|X)``.
    """

    def __init__(self, G, channel=None):
        super().__init__(channel=channel, radix=2)
        self.G = np.asarray(G, dtype=int) % 2
        self.k, self.n = self.G.shape
        self.H = _gf2.nullspace(self.G)
        self._codewords = None
        self._coset_leaders = None
        self._info_set = None
        self._G_inv = None

    # ── basic parameters ─────────────────────────────────────────────────

    @property
    def length(self):
        """The block length ``n``."""
        return self.n

    @property
    def dimension(self):
        """The number of information bits ``k``."""
        return self.k

    @property
    def message_length(self):
        """The number of message bits per codeword (``k``)."""
        return self.k

    def rate(self):
        """The code rate ``k / n``."""
        return self.k / self.n

    def __repr__(self):
        try:
            d = self.minimum_distance()
            return f"LinearCode[{self.n}, {self.k}, {d}]"
        except ditException:
            return f"LinearCode[{self.n}, {self.k}]"

    # ── encoding ─────────────────────────────────────────────────────────

    def encode(self, message):
        """
        Encode a length-``k`` message into a length-``n`` codeword.
        """
        m = np.asarray(message, dtype=int) % 2
        return tuple(int(b) for b in _gf2.matvec(self.G.T, m))

    # ── derived structure (lazy) ─────────────────────────────────────────

    def _ensure_codewords(self):
        if self._codewords is not None:
            return
        if self.k > _MAX_ENUMERATION:
            raise ditException(f"Enumerating 2^{self.k} codewords is intractable.")
        self._codewords = [
            tuple(int(b) for b in _gf2.matvec(self.G.T, np.array(m, dtype=int)))
            for m in itertools.product((0, 1), repeat=self.k)
        ]

    def _ensure_inverse(self):
        if self._info_set is not None:
            return
        _, pivots = _gf2.rref(self.G)
        self._info_set = list(pivots)
        self._G_inv = _gf2.inverse(self.G[:, self._info_set])

    def _ensure_coset_leaders(self):
        if self._coset_leaders is not None:
            return
        m = self.H.shape[0]
        if m > _MAX_ENUMERATION:
            raise ditException(f"Syndrome decoding over 2^{m} syndromes is intractable.")
        leaders = {tuple([0] * m): tuple([0] * self.n)}
        weight = 1
        while len(leaders) < 2**m and weight <= self.n:
            for positions in itertools.combinations(range(self.n), weight):
                e = np.zeros(self.n, dtype=int)
                e[list(positions)] = 1
                syndrome = tuple(int(b) for b in _gf2.matvec(self.H, e))
                if syndrome not in leaders:
                    leaders[syndrome] = tuple(int(b) for b in e)
            weight += 1
        self._coset_leaders = leaders

    def _codeword_to_message(self, codeword):
        self._ensure_inverse()
        c = np.asarray(codeword, dtype=int)
        m = _gf2.matvec(self._G_inv.T, c[self._info_set])
        return tuple(int(b) for b in m)

    # ── decoding ─────────────────────────────────────────────────────────

    def decode(self, received, channel=None):
        """
        Decode a received word.

        With no channel, hard-decision syndrome decoding is used. With a channel,
        maximum-likelihood decoding over the codebook is used.
        """
        if channel is None:
            return self._syndrome_decode(received)
        return self._ml_decode(received, channel)

    def _syndrome_decode(self, received):
        self._ensure_coset_leaders()
        y = np.asarray(received, dtype=int) % 2
        syndrome = tuple(int(b) for b in _gf2.matvec(self.H, y))
        error = np.asarray(self._coset_leaders[syndrome], dtype=int)
        codeword = (y - error) % 2
        return list(self._codeword_to_message(codeword))

    def _ml_decode(self, received, channel):
        from ._channel import channel_arrays

        inputs, outputs, P = channel_arrays(channel)
        in_index = {v: i for i, v in enumerate(inputs)}
        out_index = {v: i for i, v in enumerate(outputs)}
        self._ensure_codewords()
        cols = [out_index[y] for y in received]
        best, best_score = None, None
        for codeword in self._codewords:
            score = 0.0
            for i, bit in enumerate(codeword):
                p = P[in_index[bit], cols[i]]
                score += np.log(p) if p > 0 else -np.inf
            if best_score is None or score > best_score:
                best, best_score = codeword, score
        return list(self._codeword_to_message(best))

    # ── code-theoretic properties ────────────────────────────────────────

    def codewords(self):
        """The list of all codewords."""
        self._ensure_codewords()
        return list(self._codewords)

    def weight_enumerator(self):
        """A ``Counter`` mapping each codeword weight to its multiplicity."""
        self._ensure_codewords()
        return Counter(sum(c) for c in self._codewords)

    def minimum_distance(self):
        """
        The minimum distance: the least weight among nonzero codewords.
        """
        self._ensure_codewords()
        weights = [sum(c) for c in self._codewords if any(c)]
        return min(weights)

    def error_correcting_capability(self):
        """The number of errors the code can correct, ``floor((d - 1) / 2)``."""
        return (self.minimum_distance() - 1) // 2

    @property
    def generator_matrix(self):
        """The generator matrix ``G``."""
        return self.G

    @property
    def parity_check_matrix(self):
        """The parity-check matrix ``H``."""
        return self.H
