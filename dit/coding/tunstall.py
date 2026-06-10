"""
Tunstall coding: a variable-to-fixed-length source code.

Where a symbol code (e.g. Huffman) maps each source symbol to a variable-length
codeword, a Tunstall code parses the source into variable-length *words* drawn
from a dictionary and maps each word to a fixed-length codeword. The dictionary
is the set of leaves of a complete parse tree grown by repeatedly expanding the
most probable leaf (Tunstall, 1967).
"""

import heapq
import itertools

from ..exceptions import ditException
from ._util import DIGITS, check_radix, linear_outcomes_probs
from .base import SourceCoding

__all__ = (
    "TunstallCode",
    "tunstall",
)


def _fixed_radix(value, length, radix):
    """The fixed-`length` base-`radix` representation of ``value``."""
    digits = []
    for _ in range(length):
        value, d = divmod(value, radix)
        digits.append(DIGITS[d])
    return "".join(reversed(digits))


class TunstallCode(SourceCoding):
    """
    A variable-to-fixed-length source code.

    Parameters
    ----------
    word_to_code : dict
        A mapping from source words (tuples of outcomes) to fixed-length
        codeword strings.
    word_probs : dict
        A mapping from source words to their probabilities (used for the rate).
    code_length : int
        The fixed codeword length, in code symbols.
    dist : Distribution, None
        The source distribution.
    radix : int
        The size of the code alphabet. Default is 2.
    """

    def __init__(self, word_to_code, word_probs, code_length, dist=None, radix=2):
        super().__init__(dist=dist, radix=radix)
        self.word_to_code = dict(word_to_code)
        self.code_to_word = {code: word for word, code in self.word_to_code.items()}
        self.word_probs = dict(word_probs)
        self.code_length = code_length

    def __repr__(self):
        rows = sorted(self.word_to_code.items(), key=lambda kv: kv[1])
        lines = ["word        p        codeword"]
        for word, code in rows:
            p = self.word_probs.get(word, 0.0)
            lines.append(f"{word!r:<10}  {p:<7.4f}  {code}")
        return "\n".join(lines)

    def expected_word_length(self):
        """
        The expected number of source symbols per dictionary word.

        Returns
        -------
        L : float
        """
        return sum(p * len(word) for word, p in self.word_probs.items())

    def rate(self):
        """
        The expected number of code symbols per source symbol.

        Each word maps to ``code_length`` code symbols and spans
        ``expected_word_length`` source symbols on average.
        """
        return self.code_length / self.expected_word_length()

    def encode(self, source):
        """
        Encode a sequence of source outcomes by greedily parsing it into words.

        Parameters
        ----------
        source : iterable
            A sequence of source outcomes whose symbols form complete words.

        Returns
        -------
        encoded : str
        """
        out = []
        current = ()
        for symbol in source:
            current = current + (symbol,)
            code = self.word_to_code.get(current)
            if code is not None:
                out.append(code)
                current = ()
        if current:
            raise ditException("The source ends mid-word; cannot encode a partial Tunstall word.")
        return "".join(out)

    def decode(self, encoded):
        """
        Decode a string of code symbols back into source outcomes.

        Parameters
        ----------
        encoded : str
            A concatenation of fixed-length codewords.

        Returns
        -------
        source : list
        """
        if len(encoded) % self.code_length != 0:
            raise ditException("The encoded length is not a multiple of the code length.")
        out = []
        for i in range(0, len(encoded), self.code_length):
            block = encoded[i : i + self.code_length]
            try:
                out.extend(self.code_to_word[block])
            except KeyError:
                raise ditException(f"{block!r} is not a Tunstall codeword.") from None
        return out


def tunstall(dist, code_length, radix=2):
    """
    Build a Tunstall code for a memoryless source.

    Parameters
    ----------
    dist : Distribution
        The source distribution over single symbols (assumed i.i.d.).
    code_length : int
        The fixed codeword length, in code symbols. The dictionary holds up to
        ``radix ** code_length`` words.
    radix : int
        The size of the code alphabet. Default is 2.

    Returns
    -------
    code : TunstallCode
    """
    check_radix(radix)
    outcomes, probs = linear_outcomes_probs(dist)
    alphabet = list(zip(outcomes, probs, strict=True))
    capacity = radix**code_length
    if len(alphabet) > capacity:
        raise ditException(
            f"code_length={code_length} gives only {capacity} codewords, too few for a {len(alphabet)}-symbol source."
        )

    counter = itertools.count()
    # Max-heap on probability via negation; leaves are (word, prob).
    heap = [(-p, next(counter), (outcome,), p) for outcome, p in alphabet]
    heapq.heapify(heap)
    leaves = {(outcome,): p for outcome, p in alphabet}
    expansion = len(alphabet) - 1

    while len(leaves) + expansion <= capacity:
        _, _, word, prob = heapq.heappop(heap)
        del leaves[word]
        for outcome, p in alphabet:
            child = word + (outcome,)
            child_prob = prob * p
            leaves[child] = child_prob
            heapq.heappush(heap, (-child_prob, next(counter), child, child_prob))

    ordered = sorted(leaves, key=lambda w: (-leaves[w], tuple(map(repr, w))))
    word_to_code = {word: _fixed_radix(i, code_length, radix) for i, word in enumerate(ordered)}
    return TunstallCode(word_to_code, leaves, code_length, dist=dist, radix=radix)
