"""
Tests for dit.utils.table.
"""

from dit.params import ditParams
from dit.utils.table import build_table


def test_build_table_default():
    table = build_table(field_names=["a", "b"], title="T")
    assert table.field_names == ["a", "b"]


def test_build_table_linechar_style():
    old = ditParams["text.font"]
    try:
        ditParams["text.font"] = "linechar"
        table = build_table(field_names=["a", "b"])
        assert table.field_names == ["a", "b"]
    finally:
        ditParams["text.font"] = old
