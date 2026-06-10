"""
Tests for dit.coding symbol codes.
"""

import pytest

from dit import Distribution as D
from dit.coding import (
    SymbolCode,
    fano,
    huffman,
    length_limited_huffman,
    shannon,
    shannon_fano_elias,
)
from dit.exceptions import ditException

CONSTRUCTORS = [shannon, fano, shannon_fano_elias, huffman]


def uniform(n):
    """A uniform distribution over n integer outcomes."""
    return D(list(range(n)), [1 / n] * n)


def skewed():
    """A fixed skewed distribution."""
    return D(list(range(5)), [0.4, 0.2, 0.2, 0.1, 0.1])


@pytest.mark.parametrize("construct", CONSTRUCTORS)
def test_roundtrip(construct):
    """Every symbol code recovers the source sequence."""
    d = skewed()
    code = construct(d)
    outcomes = d.outcomes
    seq = [outcomes[i] for i in [0, 1, 2, 3, 4, 0, 1, 0, 4, 2]]
    assert code.decode(code.encode(seq)) == seq


@pytest.mark.parametrize("construct", CONSTRUCTORS)
def test_prefix_free_and_uniquely_decodable(construct):
    """Every symbol code constructed here is prefix-free (hence UD)."""
    code = construct(skewed())
    assert code.is_prefix_free()
    assert code.is_uniquely_decodable()


@pytest.mark.parametrize("construct", CONSTRUCTORS)
def test_kraft_inequality(construct):
    """The Kraft sum is at most 1."""
    code = construct(skewed())
    assert code.kraft_sum() <= 1 + 1e-9


@pytest.mark.parametrize("n", range(2, 9))
def test_shannon_entropy_bound(n):
    """Shannon coding satisfies H <= L < H + 1."""
    d = uniform(n)
    code = shannon(d)
    H = code.source_entropy()
    L = code.average_length()
    assert H <= L + 1e-9
    assert L < H + 1


def test_huffman_optimal():
    """Huffman is optimal; Shannon and Elias are not better."""
    d = skewed()
    h = huffman(d)
    assert h.is_optimal()
    assert h.average_length() <= shannon(d).average_length() + 1e-9
    assert h.average_length() <= shannon_fano_elias(d).average_length() + 1e-9


def test_huffman_is_complete():
    """A binary Huffman code is complete."""
    assert huffman(skewed()).is_complete()


def test_huffman_cover_thomas_5_1():
    """Cover & Thomas Example 5.1 pins exact Huffman lengths and length 2.3."""
    d = D(list(range(5)), [0.25, 0.25, 0.2, 0.15, 0.15])
    h = huffman(d)
    lengths = sorted(len(w) for w in h.codebook.values())
    assert lengths == [2, 2, 2, 3, 3]
    assert h.average_length() == pytest.approx(2.3)


@pytest.mark.parametrize("radix", [2, 3, 4])
def test_huffman_radix(radix):
    """D-ary Huffman is prefix-free and complete-ish under Kraft."""
    d = D(list(range(7)), [0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05])
    h = huffman(d, radix=radix)
    assert h.is_prefix_free()
    assert h.kraft_sum() <= 1 + 1e-9


def test_radix_efficiency():
    """Efficiency is the ratio of entropy to rate, in (0, 1]."""
    code = huffman(skewed())
    assert code.efficiency() == pytest.approx(code.source_entropy() / code.rate())
    assert 0 < code.efficiency() <= 1 + 1e-9


def test_length_limited_respects_bound():
    """Length-limited Huffman never exceeds the max length."""
    d = D(list(range(8)), [0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02])
    code = length_limited_huffman(d, max_length=4)
    assert max(len(w) for w in code.codebook.values()) <= 4
    assert code.is_prefix_free()


def test_length_limited_matches_huffman_when_unconstrained():
    """With a generous bound, length-limited Huffman equals Huffman."""
    d = skewed()
    relaxed = length_limited_huffman(d, max_length=10)
    assert relaxed.average_length() == pytest.approx(huffman(d).average_length())


def test_length_limited_infeasible():
    """Too small a bound is rejected."""
    d = uniform(8)
    with pytest.raises(ditException):
        length_limited_huffman(d, max_length=2)


@pytest.mark.parametrize("construct", CONSTRUCTORS)
def test_single_outcome(construct):
    """A degenerate one-outcome source gets a length-1 codeword."""
    d = D([0], [1.0])
    code = construct(d)
    assert len(code.codebook) == 1
    word = next(iter(code.codebook.values()))
    assert len(word) == 1
    out = d.outcomes
    assert code.decode(code.encode([out[0], out[0]])) == [out[0], out[0]]


@pytest.mark.parametrize("construct", CONSTRUCTORS)
def test_log_base_distribution(construct):
    """Codes built from a log-space distribution match the linear ones."""
    d = skewed()
    dlog = d.copy()
    dlog.set_base(2)
    code = construct(dlog)
    assert code.average_length() == pytest.approx(construct(d).average_length())


def test_sardinas_patterson():
    """Sardinas-Patterson distinguishes UD from non-UD codes."""
    # Uniquely decodable but not prefix-free.
    ud = SymbolCode({"a": "0", "b": "01", "c": "011"})
    assert ud.is_uniquely_decodable()
    assert not ud.is_prefix_free()
    # A classic non-uniquely-decodable code.
    not_ud = SymbolCode({"a": "0", "b": "010", "c": "01", "d": "10"})
    assert not not_ud.is_uniquely_decodable()


def test_singular_code_rejected():
    """Repeated codewords make the code singular."""
    with pytest.raises(ditException):
        SymbolCode({"a": "0", "b": "0"})


def test_decode_requires_prefix_free():
    """Decoding a non-prefix-free code raises."""
    code = SymbolCode({"a": "0", "b": "01", "c": "011"})
    with pytest.raises(ditException):
        code.decode("0011")


def test_encode_unknown_outcome():
    """Encoding an unknown outcome raises."""
    code = huffman(skewed())
    with pytest.raises(ditException):
        code.encode(["not-an-outcome"])
