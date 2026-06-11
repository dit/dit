"""
Symbol-code constructors.

Each function takes a source :class:`~dit.Distribution` and returns a
:class:`SymbolCode` built by the named algorithm.

References
----------
Shannon, Fano, Shannon-Fano-Elias, Huffman, and the Kraft inequality follow
Cover & Thomas, *Elements of Information Theory*, Ch. 5. Length-limited Huffman
uses the package-merge algorithm (Larmore & Hirschberg, 1990); Golomb codes
follow Golomb (1966).
"""

import heapq
import itertools
from math import ceil, floor, log, log2

from ..exceptions import ditException
from ._util import DIGITS, check_radix, linear_outcomes_probs, radix_expansion
from .symbol_code import SymbolCode

__all__ = (
    "fano",
    "golomb",
    "huffman",
    "length_limited_huffman",
    "rice",
    "shannon",
    "shannon_fano_elias",
)


def _sorted_pairs(dist):
    """Return ``(outcome, prob)`` pairs sorted by decreasing probability."""
    outcomes, probs = linear_outcomes_probs(dist)
    pairs = list(zip(outcomes, probs, strict=True))
    # Sort by decreasing probability; break ties by outcome repr for determinism.
    pairs.sort(key=lambda op: (-op[1], repr(op[0])))
    return pairs


def shannon(dist, radix=2):
    """
    Build a Shannon code.

    Codeword lengths are ``ceil(log_radix(1/p))`` and codewords are read off the
    base-`radix` expansion of the cumulative distribution (outcomes sorted by
    decreasing probability).

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    radix : int
        The size of the code alphabet. Default is 2.

    Returns
    -------
    code : SymbolCode
    """
    check_radix(radix)
    pairs = _sorted_pairs(dist)
    if len(pairs) == 1:
        return SymbolCode({pairs[0][0]: DIGITS[0]}, dist=dist, radix=radix)

    codebook = {}
    cumulative = 0.0
    for outcome, p in pairs:
        length = max(1, ceil(-log(p) / log(radix)))
        codebook[outcome] = radix_expansion(cumulative, length, radix)
        cumulative += p
    return SymbolCode(codebook, dist=dist, radix=radix)


def shannon_fano_elias(dist, radix=2):
    """
    Build a Shannon-Fano-Elias code.

    Codeword lengths are ``ceil(log_radix(1/p)) + 1`` and codewords are read off
    the base-`radix` expansion of the *midpoint* cumulative distribution
    ``F_bar(x) = sum_{y<x} p(y) + p(x)/2``. The result is prefix-free.

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    radix : int
        The size of the code alphabet. Default is 2.

    Returns
    -------
    code : SymbolCode
    """
    check_radix(radix)
    pairs = _sorted_pairs(dist)

    codebook = {}
    cumulative = 0.0
    for outcome, p in pairs:
        length = ceil(-log(p) / log(radix)) + 1
        midpoint = cumulative + p / 2
        codebook[outcome] = radix_expansion(midpoint, length, radix)
        cumulative += p
    return SymbolCode(codebook, dist=dist, radix=radix)


def fano(dist, radix=2):
    """
    Build a Fano code (the Shannon-Fano top-down splitting method).

    Outcomes are sorted by probability and recursively partitioned into two
    groups of as-equal-as-possible total probability; one bit distinguishes the
    groups at each level.

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    radix : int
        The size of the code alphabet. Only ``radix=2`` is currently supported.

    Returns
    -------
    code : SymbolCode
    """
    check_radix(radix)
    if radix != 2:
        raise ditException("Fano coding currently supports only binary codes (radix=2).")
    pairs = _sorted_pairs(dist)
    if len(pairs) == 1:
        return SymbolCode({pairs[0][0]: DIGITS[0]}, dist=dist, radix=radix)

    codebook = {outcome: "" for outcome, _ in pairs}

    def split(items):
        if len(items) == 1:
            return
        total = sum(p for _, p in items)
        # Find the split point minimizing the imbalance between the two halves.
        best_index, best_diff, left = 1, None, 0.0
        for i in range(1, len(items)):
            left += items[i - 1][1]
            diff = abs(total - 2 * left)
            if best_diff is None or diff < best_diff:
                best_diff, best_index = diff, i
        for outcome, _ in items[:best_index]:
            codebook[outcome] += "0"
        for outcome, _ in items[best_index:]:
            codebook[outcome] += "1"
        split(items[:best_index])
        split(items[best_index:])

    split(pairs)
    return SymbolCode(codebook, dist=dist, radix=radix)


def huffman(dist, radix=2):
    """
    Build a Huffman code, the optimal symbol code for the source.

    For ``radix > 2`` the alphabet is padded with zero-probability dummy symbols
    so that every internal node has exactly `radix` children.

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    radix : int
        The size of the code alphabet. Default is 2.

    Returns
    -------
    code : SymbolCode
    """
    check_radix(radix)
    pairs = _sorted_pairs(dist)
    if len(pairs) == 1:
        return SymbolCode({pairs[0][0]: DIGITS[0]}, dist=dist, radix=radix)

    dummy = object()
    nodes = [(p, ("leaf", outcome)) for outcome, p in pairs]
    if radix > 2:
        while (len(nodes) - 1) % (radix - 1) != 0:
            nodes.append((0.0, ("leaf", dummy)))

    counter = itertools.count()
    heap = [(p, next(counter), node) for p, node in nodes]
    heapq.heapify(heap)
    while len(heap) > 1:
        children = [heapq.heappop(heap) for _ in range(min(radix, len(heap)))]
        weight = sum(c[0] for c in children)
        node = ("internal", [c[2] for c in children])
        heapq.heappush(heap, (weight, next(counter), node))

    codebook = {}

    def assign(node, prefix):
        kind, payload = node
        if kind == "leaf":
            if payload is not dummy:
                codebook[payload] = prefix
        else:
            for digit, child in enumerate(payload):
                assign(child, prefix + DIGITS[digit])

    assign(heap[0][2], "")
    return SymbolCode(codebook, dist=dist, radix=radix)


def length_limited_huffman(dist, max_length, radix=2):
    """
    Build an optimal prefix code whose codewords are at most `max_length` long.

    Uses the package-merge algorithm (Larmore & Hirschberg, 1990).

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    max_length : int
        The maximum allowed codeword length.
    radix : int
        The size of the code alphabet. Only ``radix=2`` is currently supported.

    Returns
    -------
    code : SymbolCode
    """
    check_radix(radix)
    if radix != 2:
        raise ditException("Length-limited Huffman coding currently supports only binary codes (radix=2).")
    pairs = _sorted_pairs(dist)
    n = len(pairs)
    if n == 1:
        return SymbolCode({pairs[0][0]: DIGITS[0]}, dist=dist, radix=radix)
    if 2**max_length < n:
        raise ditException(f"max_length={max_length} is too small to code {n} symbols.")

    weights = [p for _, p in pairs]
    lengths = _package_merge(weights, max_length)
    codebook = _canonical_codewords([outcome for outcome, _ in pairs], lengths)
    return SymbolCode(codebook, dist=dist, radix=radix)


def _package_merge(weights, max_length):
    """
    Compute optimal length-limited codeword lengths via package-merge.

    Parameters
    ----------
    weights : list of float
        The symbol probabilities.
    max_length : int
        The maximum codeword length.

    Returns
    -------
    lengths : list of int
        ``lengths[i]`` is the codeword length for symbol ``i``.
    """
    n = len(weights)
    coins = sorted(((w, (i,)) for i, w in enumerate(weights)), key=lambda c: c[0])
    packages = list(coins)
    for _ in range(max_length - 1):
        paired = [
            (packages[i][0] + packages[i + 1][0], packages[i][1] + packages[i + 1][1])
            for i in range(0, len(packages) - 1, 2)
        ]
        packages = sorted(coins + paired, key=lambda c: c[0])

    lengths = [0] * n
    for _, indices in packages[: 2 * (n - 1)]:
        for i in indices:
            lengths[i] += 1
    return lengths


def _canonical_codewords(outcomes, lengths):
    """
    Assign canonical prefix-free codewords to outcomes given their lengths.

    Parameters
    ----------
    outcomes : list
        The source outcomes.
    lengths : list of int
        The codeword length for each outcome.

    Returns
    -------
    codebook : dict
    """
    order = sorted(range(len(outcomes)), key=lambda i: (lengths[i], repr(outcomes[i])))
    codebook = {}
    code = 0
    prev_length = None
    for i in order:
        length = lengths[i]
        if prev_length is not None:
            code = (code + 1) << (length - prev_length)
        codebook[outcomes[i]] = format(code, f"0{length}b")
        prev_length = length
    return codebook


def _truncated_binary(value, m):
    """The truncated binary codeword for ``value`` in ``[0, m)``."""
    b = floor(log2(m)) if m > 1 else 0
    cutoff = (1 << (b + 1)) - m
    if value < cutoff:
        return format(value, f"0{b}b") if b > 0 else ""
    return format(value + cutoff, f"0{b + 1}b")


def _golomb_codeword(n, m):
    """The Golomb codeword for non-negative integer ``n`` with parameter ``m``."""
    quotient, remainder = divmod(n, m)
    return "1" * quotient + "0" + _truncated_binary(remainder, m)


def _optimal_golomb_m(dist):
    """The optimal Golomb parameter ``m`` for a (geometric) source."""
    outcomes, probs = linear_outcomes_probs(dist)
    mean = sum(o * p for o, p in zip(outcomes, probs, strict=True))
    if mean <= 0:
        return 1
    # theta is the geometric ratio implied by the mean: E[n] = theta / (1 - theta).
    theta = mean / (1 + mean)
    if theta <= 0 or theta >= 1:
        return 1
    return max(1, ceil(-1 / log2(theta)))


def golomb(dist, m=None, radix=2):
    """
    Build a Golomb code over non-negative-integer outcomes.

    Each codeword is the unary-coded quotient ``n // m`` followed by the
    truncated-binary remainder ``n % m``. Golomb codes are optimal prefix codes
    for geometrically distributed sources.

    Parameters
    ----------
    dist : Distribution
        The source distribution; its outcomes must be non-negative integers.
    m : int, None
        The Golomb parameter. If None, the parameter optimal for the (assumed
        geometric) source is chosen from its mean.
    radix : int
        The size of the code alphabet. Only ``radix=2`` is currently supported.

    Returns
    -------
    code : SymbolCode
    """
    check_radix(radix)
    if radix != 2:
        raise ditException("Golomb coding is binary (radix=2).")
    outcomes, _ = linear_outcomes_probs(dist)
    if not all(isinstance(o, int) and o >= 0 for o in outcomes):
        raise ditException("Golomb coding requires non-negative integer outcomes.")
    if m is None:
        m = _optimal_golomb_m(dist)
    if m < 1:
        raise ditException(f"The Golomb parameter m must be >= 1, got {m!r}.")
    codebook = {o: _golomb_codeword(o, m) for o in outcomes}
    return SymbolCode(codebook, dist=dist, radix=radix)


def rice(dist, k=None):
    """
    Build a Rice code, the Golomb code with parameter ``m = 2 ** k``.

    Parameters
    ----------
    dist : Distribution
        The source distribution; its outcomes must be non-negative integers.
    k : int, None
        The Rice parameter. If None, it is chosen from the optimal Golomb
        parameter for the source.

    Returns
    -------
    code : SymbolCode
    """
    if k is None:
        k = max(0, round(log2(_optimal_golomb_m(dist))))
    if k < 0:
        raise ditException(f"The Rice parameter k must be >= 0, got {k!r}.")
    return golomb(dist, m=2**k)
