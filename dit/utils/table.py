"""
"""

try:
    from prettytable import UNICODE_LINES, PrettyTable
except ImportError:
    from pltable import UNICODE_LINES, PrettyTable

from ..import ditParams


__all__ = (
    'build_table',
)


def build_table(field_names=None, title=None):
    """
    Construct a PrettyTable with style set.

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
    if ditParams['text.font'] == 'linechar':
        try:
            table.set_style(UNICODE_LINES)
        except Exception:
            pass
    return table
