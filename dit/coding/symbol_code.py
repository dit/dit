"""
Symbol codes: one codeword per source outcome.

A :class:`SymbolCode` is the codebook-based realization of a source code, where
each source outcome is mapped to a single codeword. This covers Shannon, Fano,
Shannon-Fano-Elias, Huffman (and its length-limited variant), Golomb/Rice, and
the universal integer codes.
"""

from math import isclose

from ..exceptions import ditException
from ._util import linear_outcomes_probs
from .base import SourceCoding

__all__ = ("SymbolCode",)


class SymbolCode(SourceCoding):
    """
    A source code that maps each outcome to a single codeword.

    Parameters
    ----------
    codebook : dict
        A mapping from source outcomes to codeword strings (e.g. ``'010'``).
    dist : Distribution, None
        The source distribution. Required for the rate-based properties
        (:meth:`rate`, :meth:`average_length`, :meth:`redundancy`, ...).
    radix : int
        The size of the code alphabet. Default is 2 (binary).
    """

    def __init__(self, codebook, dist=None, radix=2):
        super().__init__(dist=dist, radix=radix)
        self.codebook = dict(codebook)
        if len(set(self.codebook.values())) != len(self.codebook):
            raise ditException("Codewords must be distinct (the code is singular).")
        self._trie = None

    # ── representation ───────────────────────────────────────────────────

    def __repr__(self):
        probs = self._prob_dict()
        rows = sorted(self.codebook.items(), key=lambda kv: (len(kv[1]), kv[1]))
        width = max((len(repr(o)) for o in self.codebook), default=7)
        lines = [f"{'outcome':>{width}}   p        codeword"]
        for outcome, word in rows:
            p = probs.get(outcome)
            p_str = f"{p:<7.4f}" if p is not None else "  -    "
            lines.append(f"{outcome!r:>{width}}   {p_str}  {word}")
        return "\n".join(lines)

    # ── helpers ──────────────────────────────────────────────────────────

    def _prob_dict(self):
        """A dict mapping each outcome to its (linear) probability."""
        if self.dist is None:
            return {}
        outcomes, probs = linear_outcomes_probs(self.dist)
        return dict(zip(outcomes, probs, strict=True))

    def _build_trie(self):
        """Build a decoding trie; leaves carry the outcome under the ``''`` key."""
        if not self.is_prefix_free():
            raise ditException("Decoding is only supported for prefix-free codes.")
        trie = {}
        for outcome, word in self.codebook.items():
            node = trie
            for symbol in word:
                node = node.setdefault(symbol, {})
            node[""] = outcome
        return trie

    # ── encode / decode ──────────────────────────────────────────────────

    def encode(self, source):
        """
        Encode a sequence of source outcomes into a string of code symbols.

        Parameters
        ----------
        source : iterable
            A sequence of source outcomes.

        Returns
        -------
        encoded : str
            The concatenated codewords.
        """
        try:
            return "".join(self.codebook[outcome] for outcome in source)
        except KeyError as e:
            raise ditException(f"Outcome {e.args[0]!r} is not in the codebook.") from None

    def decode(self, encoded):
        """
        Decode a string of code symbols back into source outcomes.

        Parameters
        ----------
        encoded : str
            A concatenation of codewords.

        Returns
        -------
        source : list
            The decoded sequence of source outcomes.
        """
        if self._trie is None:
            self._trie = self._build_trie()
        source = []
        node = self._trie
        for symbol in encoded:
            try:
                node = node[symbol]
            except KeyError:
                raise ditException(f"Code symbol {symbol!r} is not part of any codeword.") from None
            if "" in node:
                source.append(node[""])
                node = self._trie
        if node is not self._trie:
            raise ditException("The encoded string is not a valid sequence of codewords.")
        return source

    # ── rate-based properties ────────────────────────────────────────────

    def average_length(self):
        """
        The expected codeword length, ``sum_x p(x) * len(codeword(x))``.

        Returns
        -------
        L : float
        """
        probs = self._prob_dict()
        if not probs:
            raise ditException("A source distribution is required to compute the average length.")
        return sum(p * len(self.codebook[outcome]) for outcome, p in probs.items())

    def rate(self):
        """
        The expected number of code symbols per source symbol.

        For a symbol code this is exactly the average codeword length.
        """
        return self.average_length()

    def length_variance(self):
        """
        The variance of the codeword length under the source distribution.

        Among optimal codes (which share the minimal average length) one often
        prefers the one of least length variance.

        Returns
        -------
        var : float
        """
        probs = self._prob_dict()
        if not probs:
            raise ditException("A source distribution is required to compute the length variance.")
        mean = self.average_length()
        return sum(p * (len(self.codebook[outcome]) - mean) ** 2 for outcome, p in probs.items())

    # ── structural properties ────────────────────────────────────────────

    def kraft_sum(self):
        """
        The Kraft sum ``sum_x radix ** -len(codeword(x))``.

        By the Kraft-McMillan inequality this is ``<= 1`` for any uniquely
        decodable code, with equality iff the code is complete.

        Returns
        -------
        K : float
        """
        return sum(self.radix ** -len(word) for word in self.codebook.values())

    def is_complete(self):
        """
        Whether the code is complete (its Kraft sum equals 1).

        Returns
        -------
        complete : bool
        """
        return isclose(self.kraft_sum(), 1.0, abs_tol=1e-9)

    def is_prefix_free(self):
        """
        Whether no codeword is a prefix of another (the code is instantaneous).

        Returns
        -------
        prefix_free : bool
        """
        words = sorted(self.codebook.values(), key=len)
        seen = []
        for word in words:
            if any(word.startswith(s) for s in seen):
                return False
            seen.append(word)
        return True

    def is_uniquely_decodable(self):
        """
        Whether the code is uniquely decodable, via the Sardinas-Patterson test.

        Returns
        -------
        unique : bool
        """
        words = set(self.codebook.values())
        if "" in words:
            return False

        def dangling(c1, c2):
            """Suffixes left after matching codewords of ``c1`` against ``c2``."""
            suffixes = set()
            for a in c1:
                for b in c2:
                    if a == b:
                        continue
                    if a.startswith(b):
                        suffixes.add(a[len(b) :])
                    elif b.startswith(a):
                        suffixes.add(b[len(a) :])
            return suffixes

        # The first dangling set comes from matching codewords against each other.
        current = dangling(words, words)
        seen = set()
        while current:
            if words & current:
                # A codeword is itself a dangling suffix => not uniquely decodable.
                return False
            if current <= seen:
                break
            seen |= current
            current = dangling(words, current)
        return True

    def is_optimal(self):
        """
        Whether the code achieves the minimal average length (Huffman optimal).

        Returns
        -------
        optimal : bool
        """
        if self.dist is None:
            raise ditException("A source distribution is required to test optimality.")
        from .codes import huffman

        best = huffman(self.dist, radix=self.radix).average_length()
        return isclose(self.average_length(), best, abs_tol=1e-9)
