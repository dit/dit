"""
Tests for the dit.coding base classes.
"""

import pytest

from dit import Distribution as D
from dit.coding import ChannelCoding, SourceCoding, huffman
from dit.exceptions import ditException


def test_source_coding_is_abstract():
    """SourceCoding cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SourceCoding()


def test_channel_coding_is_abstract():
    """ChannelCoding cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ChannelCoding()


def test_symbol_code_is_source_coding():
    """A built symbol code is a SourceCoding instance."""
    code = huffman(D(list(range(3)), [0.5, 0.3, 0.2]))
    assert isinstance(code, SourceCoding)


def test_redundancy_and_efficiency_consistent():
    """redundancy = rate - entropy and efficiency = entropy / rate."""
    code = huffman(D(list(range(5)), [0.4, 0.2, 0.2, 0.1, 0.1]))
    assert code.redundancy() == pytest.approx(code.rate() - code.source_entropy())
    assert code.efficiency() == pytest.approx(code.source_entropy() / code.rate())


def test_properties_require_distribution():
    """Rate-based properties need a source distribution."""
    from dit.coding import SymbolCode

    code = SymbolCode({"a": "0", "b": "1"})
    with pytest.raises(ditException):
        code.average_length()
    with pytest.raises(ditException):
        code.source_entropy()
