""" """

import contextlib

try:
    from pltable import UNICODE_LINES, PrettyTable

    _LINE_STYLE = UNICODE_LINES
except ImportError:
    from prettytable import PrettyTable, TableStyle

    _LINE_STYLE = TableStyle.SINGLE_BORDER

from ..params import ditParams

__all__ = ("build_table",)


def build_table(field_names=None, title=None):
    """
    Construct a PrettyTable with style set.

    When ``ditParams["text.font"]`` is ``"linechar"``, the table is drawn with
    double-line borders if PLTable is installed, and with prettytable's
    single-line borders otherwise.

    Parameters
    ----------
    field_names : list
        List of field names.
    title : str
        Title for the table.

    Returns
    -------
    table : PrettyTable
        A new table.
    """
    table = PrettyTable(field_names=field_names, title=title)
    if ditParams["text.font"] == "linechar":
        with contextlib.suppress(Exception):
            table.set_style(_LINE_STYLE)
    return table
