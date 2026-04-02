"""
Helper functions related to partial information decompositions.
"""

import numpy as np

from ..utils import build_table
from .measures import __all_pids
from .pid import sort_key

__all__ = ("compare_measures", "pointwise_pid_table")


def compare_measures(dist, pids=__all_pids, inputs=None, output=None, name="", digits=5):
    """
    Print the results of several partial information decompositions.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to compute PIDs of.

    pids : iterable(BasePID)
        The PID classes to use.

    inputs : iterable of iterables
        The variables to treat as inputs.

    output : iterable
        The variables to consider as a singular output.

    name : str
        The name of distribution.

    digits : int
        The number of digits of precision to print.
    """
    pids = [pid(dist.copy(), inputs, output) for pid in pids]
    names = [pid.name for pid in pids]
    dist_name = getattr(dist, "name", "")
    title = f"PID Comparison | {dist_name}" if dist_name else "PID Comparison"
    table = build_table(field_names=([name] + names), title=title)
    for name in names:
        table.float_format[name] = f" {digits + 2}.{digits}"
    nodes = sorted(pids[0]._lattice, key=sort_key(pids[0]._lattice))
    stringify = lambda node: "".join("{{{}}}".format(":".join(map(str, n))) for n in node)
    for node in nodes:
        vals = [pid[node] for pid in pids]
        vals = [0.0 if np.isclose(0, val, atol=1e-5, rtol=1e-5) else val for val in vals]
        table.add_row([stringify(node)] + vals)
    print(table.get_string())


def pointwise_pid_table(dist, sources=None, target=None, pid_class=None, digits=4):
    """
    Print a table of per-outcome pointwise PID values.

    Each row corresponds to one joint outcome.  Columns are:
    ``event``, ``p``, then for each lattice node (sorted by depth):
    ``{node} pi`` and — when available — ``{node} pi+`` and ``{node} pi-``.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to analyse.
    sources : iter of iters, optional
        Source variable indices.  Defaults to all variables except the last.
    target : iter, optional
        Target variable indices.  Defaults to the last variable.
    pid_class : BasePointwisePID subclass, optional
        The pointwise PID measure to use.  Defaults to ``PID_SX``.
    digits : int
        Number of decimal places to display.
    """
    if pid_class is None:
        from .measures.isx import PID_SX
        pid_class = PID_SX

    pid = pid_class(dist, sources, target, pointwise=True)
    has_parts = bool(pid._pw_pis_plus)

    nodes = sorted(pid._lattice, key=sort_key(pid._lattice))
    node_labels = ["".join("{{{}}}".format(":".join(map(str, n))) for n in node) for node in nodes]

    node_columns = []
    for label in node_labels:
        node_columns.append(f"{label} pi")
        if has_parts:
            node_columns += [f"{label} pi+", f"{label} pi-"]

    columns = ["event", "p"] + node_columns
    dist_name = getattr(dist, "name", "")
    base_title = f"{pid._name} (pointwise)"
    title = f"{base_title} | {dist_name}" if dist_name else base_title
    table = build_table(columns, title=title)
    for col in columns[1:]:
        table.float_format[col] = f"{digits + 2}.{digits}"

    linear_probs = {
        o: dist.ops.exp(dist[o]) if dist.is_log() else dist[o]
        for o in dist.outcomes
    }

    for outcome in dist.outcomes:
        p = linear_probs[outcome]
        row = [str(outcome), p]
        for node in nodes:
            row.append(pid._pw_pis[node][outcome])
            if has_parts:
                row.append(pid._pw_pis_plus[node][outcome])
                row.append(pid._pw_pis_minus[node][outcome])
        table.add_row(row)

    print(table.get_string())
