"""
Tests for dit.utils.table.
"""

import importlib
import sys

import pytest

import dit.utils.table as table_module
from dit.params import ditParams
from dit.utils.table import build_table


@pytest.fixture
def reload_table(monkeypatch):
    """Reload dit.utils.table, optionally with PLTable hidden; restore it afterwards."""

    def reload(hide_pltable=False):
        if hide_pltable:
            monkeypatch.setitem(sys.modules, "pltable", None)
        return importlib.reload(table_module)

    yield reload
    monkeypatch.undo()
    importlib.reload(table_module)


def _render(module):
    table = module.build_table(field_names=["a", "b"], title="T")
    table.add_row(["0", 0.5])
    return table.get_string()


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


def test_linechar_double_lines_with_pltable(reload_table, monkeypatch):
    pytest.importorskip("pltable")
    monkeypatch.setitem(ditParams, "text.font", "linechar")
    assert "╔" in _render(reload_table())


def test_linechar_single_lines_with_prettytable(reload_table, monkeypatch):
    monkeypatch.setitem(ditParams, "text.font", "linechar")
    rendered = _render(reload_table(hide_pltable=True))
    assert "┌" in rendered
    assert "╔" not in rendered
    assert "║" not in rendered


def test_ascii_tables_match(reload_table, monkeypatch):
    """With the default font, PLTable and prettytable draw the same table."""
    pytest.importorskip("pltable")
    monkeypatch.setitem(ditParams, "text.font", "ascii")
    assert _render(reload_table()) == _render(reload_table(hide_pltable=True))
