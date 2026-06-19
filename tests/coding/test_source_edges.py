"""
Edge-case and error-path coverage for the source-coding modules.
"""

import pytest

from dit import Distribution as D
from dit.coding import (
    fano,
    golomb,
    huffman,
    length_limited_huffman,
    rice,
    shannon,
    tunstall,
)
from dit.coding.symbol_code import SymbolCode
from dit.exceptions import ditException

# ── SymbolCode decode and property errors ──────────────────────────────────


def test_decode_unknown_symbol_raises():
    """Decoding a symbol outside the code alphabet raises."""
    code = SymbolCode({("a",): "0", ("b",): "10"})
    with pytest.raises(ditException):
        code.decode("2")


def test_decode_incomplete_raises():
    """A trailing partial codeword raises."""
    code = SymbolCode({("a",): "0", ("b",): "10"})
    with pytest.raises(ditException):
        code.decode("1")


def test_properties_require_distribution():
    """The rate-based properties need a source distribution."""
    code = SymbolCode({("a",): "0", ("b",): "10"})
    with pytest.raises(ditException):
        code.average_length()
    with pytest.raises(ditException):
        code.length_variance()
    with pytest.raises(ditException):
        code.is_optimal()


def test_length_variance_value():
    """The length variance is computed from the source distribution."""
    d = D(["a", "b"], [0.5, 0.5])
    code = SymbolCode({("a",): "0", ("b",): "1"}, dist=d)
    assert code.length_variance() == pytest.approx(0.0)


def test_uniquely_decodable_empty_codeword():
    """A code containing the empty codeword is not uniquely decodable."""
    code = SymbolCode({("a",): "", ("b",): "0"})
    assert not code.is_uniquely_decodable()


# ── radix validation and D-ary Huffman ─────────────────────────────────────


def test_invalid_radix_raises():
    """An out-of-range radix raises."""
    d = D(["a", "b"], [0.5, 0.5])
    with pytest.raises(ditException):
        shannon(d, radix=1)


def test_ternary_huffman():
    """A D-ary (radix 3) Huffman code with padding round-trips and is prefix-free."""
    # Four leaves over radix 3 forces a dummy-leaf pad ((4 - 1) % (3 - 1) != 0).
    d = D(["a", "b", "c", "d"], [0.4, 0.3, 0.2, 0.1])
    code = huffman(d, radix=3)
    assert code.is_prefix_free()
    seq = list(d.outcomes)
    assert code.decode(code.encode(seq)) == seq


def test_fano_non_binary_raises():
    """Fano coding only supports binary codes."""
    with pytest.raises(ditException):
        fano(D(["a", "b"], [0.5, 0.5]), radix=3)


def test_length_limited_non_binary_raises():
    """Length-limited Huffman only supports binary codes."""
    with pytest.raises(ditException):
        length_limited_huffman(D(["a", "b"], [0.5, 0.5]), 3, radix=3)


def test_length_limited_single_symbol():
    """A single-symbol source gets a one-symbol codeword."""
    code = length_limited_huffman(D(["a"], [1.0]), 3)
    assert code.decode(code.encode([("a",)])) == [("a",)]


# ── Golomb / Rice edges ────────────────────────────────────────────────────


def _geometric(theta, n):
    ps = [(1 - theta) * theta**k for k in range(n)]
    total = sum(ps)
    return D(list(range(n)), [p / total for p in ps])


def test_golomb_bad_parameter_raises():
    """A Golomb parameter below one raises."""
    with pytest.raises(ditException):
        golomb(_geometric(0.5, 8), m=0)


def test_golomb_non_integer_outcomes_raises():
    """Golomb coding requires non-negative integer outcomes."""
    with pytest.raises(ditException):
        golomb(D(["a", "b"], [0.5, 0.5]))


def test_golomb_degenerate_mean():
    """A point-mass-at-zero source yields the m = 1 Golomb code."""
    code = golomb(D([0], [1.0]))
    assert code.decode(code.encode([0])) == [0]


def test_golomb_non_power_of_two_parameter():
    """A non-power-of-two m exercises the long branch of the truncated-binary code."""
    d = _geometric(0.5, 8)
    code = golomb(d, m=3)
    seq = list(d.outcomes)
    assert code.decode(code.encode(seq)) == seq


def test_rice_bad_parameter_raises():
    """A negative Rice parameter raises."""
    with pytest.raises(ditException):
        rice(_geometric(0.5, 8), k=-1)


# ── Tunstall decode errors ─────────────────────────────────────────────────


def test_tunstall_bad_length_raises():
    """A code string whose length is not a multiple of the block size raises."""
    code = tunstall(D(["a", "b"], [0.9, 0.1]), 4)
    with pytest.raises(ditException):
        code.decode("1")


def test_tunstall_invalid_codeword_raises():
    """An unused fixed-length block is rejected on decode."""
    # A ternary source grows leaves by two per step, leaving a block unused at length 4.
    code = tunstall(D(["a", "b", "c"], [0.7, 0.2, 0.1]), 4)
    used = set(code.code_to_word)
    unused = [format(i, "04b") for i in range(16) if format(i, "04b") not in used]
    assert unused, "expected some unused blocks for this source"
    with pytest.raises(ditException):
        code.decode(unused[0])
