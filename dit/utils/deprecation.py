"""
Helpers for deprecating renamed keyword arguments.
"""

from debtcollector import renames

__all__ = ("renamed_kwarg",)


def renamed_kwarg(old_name, new_name):
    """
    Accept `old_name` as a deprecated alias for the keyword `new_name`.

    Parameters
    ----------
    old_name : str
        The deprecated keyword.
    new_name : str
        The keyword it was renamed to.

    Returns
    -------
    decorator : func
        A decorator forwarding `old_name` to `new_name` with a
        DeprecationWarning.
    """
    return renames.renamed_kwarg(old_name, new_name, version="2.4", removal_version="3.0", replace=True)
