"""
Low-density parity-check (LDPC) codes with belief-propagation decoding.

An :class:`LDPCCode` is a linear code defined by a sparse parity-check matrix
``H``, decoded with the sum-product algorithm on its Tanner graph (Gallager,
1962). Encoding and the algebraic properties are inherited from
:class:`~dit.coding.linear.LinearCode`.
"""

import numpy as np

from ..exceptions import ditException
from . import _gf2
from .linear import LinearCode

__all__ = (
    "LDPCCode",
    "gallager",
    "ldpc",
)


class LDPCCode(LinearCode):
    """
    A linear code decoded by belief propagation on a sparse parity-check matrix.

    Parameters
    ----------
    H : array-like
        An ``m x n`` binary parity-check matrix (typically sparse).
    channel : Distribution, None
        A default channel.
    max_iterations : int
        The maximum number of belief-propagation iterations.
    """

    def __init__(self, H, channel=None, max_iterations=50):
        H = np.asarray(H, dtype=int) % 2
        G = _gf2.nullspace(H)
        if G.shape[0] == 0:
            raise ditException("The parity-check matrix leaves no information bits.")
        super().__init__(G, channel=channel)
        self.H = H
        self.max_iterations = max_iterations
        self._check_vars = [np.flatnonzero(H[c]).tolist() for c in range(H.shape[0])]
        self._var_checks = [np.flatnonzero(H[:, v]).tolist() for v in range(H.shape[1])]

    def decode(self, received, channel=None):
        """
        Decode by belief propagation when a channel is given, else hard decoding.
        """
        if channel is None:
            return self._syndrome_decode(received)
        return self._belief_propagation(received, channel)

    def _belief_propagation(self, received, channel):
        from ._channel import log_likelihoods

        llr_map = log_likelihoods(channel)
        L = np.array([llr_map[y] for y in received], dtype=float)
        n, m = self.n, self.H.shape[0]

        # Variable-to-check and check-to-variable messages.
        M_vc = {(v, c): L[v] for v in range(n) for c in self._var_checks[v]}
        M_cv = {(c, v): 0.0 for c in range(m) for v in self._check_vars[c]}

        clip = 1 - 1e-12
        for _ in range(self.max_iterations):
            for c in range(m):
                vs = self._check_vars[c]
                taus = {v: np.tanh(np.clip(M_vc[(v, c)] / 2, -30, 30)) for v in vs}
                for v in vs:
                    product = 1.0
                    for v2 in vs:
                        if v2 != v:
                            product *= taus[v2]
                    product = np.clip(product, -clip, clip)
                    M_cv[(c, v)] = 2 * np.arctanh(product)

            total = L.copy()
            for v in range(n):
                for c in self._var_checks[v]:
                    total[v] += M_cv[(c, v)]
            xhat = (total < 0).astype(int)

            if not np.any(_gf2.matvec(self.H, xhat)):
                break

            for v in range(n):
                for c in self._var_checks[v]:
                    M_vc[(v, c)] = L[v] + sum(M_cv[(c2, v)] for c2 in self._var_checks[v] if c2 != c)

        return list(self._codeword_to_message(xhat))


def ldpc(H, channel=None, max_iterations=50):
    """
    Build an LDPC code from a parity-check matrix.

    Parameters
    ----------
    H : array-like
        An ``m x n`` binary parity-check matrix.
    channel : Distribution, None
        A default channel.
    max_iterations : int
        The maximum number of belief-propagation iterations.

    Returns
    -------
    code : LDPCCode
    """
    return LDPCCode(H, channel=channel, max_iterations=max_iterations)


def gallager(n, wc, wr, prng=None, channel=None):
    """
    Build a regular Gallager LDPC code.

    Parameters
    ----------
    n : int
        The block length; must be divisible by ``wr``.
    wc : int
        The column weight (variable degree).
    wr : int
        The row weight (check degree); ``wc < wr``.
    prng : numpy.random.Generator, None
        The random number generator used to permute the sub-bands.
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : LDPCCode
    """
    if n % wr != 0:
        raise ditException("Gallager construction requires n divisible by wr.")
    if not 0 < wc < wr:
        raise ditException("Gallager construction requires 0 < wc < wr.")
    if prng is None:
        prng = np.random.default_rng()

    band_rows = n // wr
    base = np.zeros((band_rows, n), dtype=int)
    for row in range(band_rows):
        base[row, row * wr : (row + 1) * wr] = 1

    bands = [base]
    for _ in range(wc - 1):
        permutation = prng.permutation(n)
        bands.append(base[:, permutation])
    H = np.vstack(bands)
    return LDPCCode(H, channel=channel)
