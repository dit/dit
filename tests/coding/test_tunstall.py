"""
Tests for dit.coding Tunstall (variable-to-fixed) codes.
"""

import pytest

from dit import Distribution as D
from dit.coding import TunstallCode, tunstall
from dit.exceptions import ditException


def binary_source(p=0.7):
    return D(["a", "b"], [p, 1 - p])


@pytest.mark.parametrize("code_length", [1, 2, 3, 4])
def test_leaf_count(code_length):
    """The dictionary fills up to radix ** code_length words."""
    code = tunstall(binary_source(), code_length=code_length)
    assert len(code.word_to_code) <= 2**code_length
    # Each codeword has exactly code_length symbols.
    assert all(len(w) == code_length for w in code.word_to_code.values())


def test_roundtrip():
    """A sequence of complete words is recovered exactly."""
    code = tunstall(binary_source(), code_length=3)
    source = [symbol for word in code.word_to_code for symbol in word]
    assert code.decode(code.encode(source)) == source


def test_rate_approaches_entropy():
    """Longer codewords drive the rate toward the source entropy."""
    d = binary_source(0.8)
    coarse = tunstall(d, code_length=2)
    fine = tunstall(d, code_length=6)
    H = coarse.source_entropy()
    assert fine.rate() >= H - 1e-9
    assert fine.rate() <= coarse.rate() + 1e-9


def test_partial_word_rejected():
    """A source ending mid-word cannot be encoded."""
    code = tunstall(binary_source(), code_length=3)
    # Find the longest word and feed a strict prefix of it.
    longest = max(code.word_to_code, key=len)
    if len(longest) > 1:
        with pytest.raises(ditException):
            code.encode(list(longest[:-1]))


def test_code_length_too_small():
    """An alphabet larger than the dictionary capacity is rejected."""
    d = D(list(range(4)), [0.25] * 4)
    with pytest.raises(ditException):
        tunstall(d, code_length=1)


def test_is_source_coding():
    """TunstallCode exposes the SourceCoding interface."""
    code = tunstall(binary_source(), code_length=3)
    assert isinstance(code, TunstallCode)
    assert code.redundancy() == pytest.approx(code.rate() - code.source_entropy())
