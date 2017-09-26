"""
Helper functions related to partial information decompositions.
"""

import prettytable

import numpy as np

from .. import ditParams
from .lattice import sort_key
from . import __all_pids


def compare_measures(dist, pids=__all_pids, inputs=None, output=None, name='', digits=5):
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
    table = prettytable.PrettyTable(field_names=([name] + names))
    if ditParams['text.font'] == 'linechar':
        try:
            table.set_style(prettytable.BOX_CHARS)
        except:
            pass
    for name in names:
        table.float_format[name] = ' {0}.{1}'.format(digits + 2, digits)
    nodes = sorted(pids[0]._lattice, key=sort_key(pids[0]._lattice))
    stringify = lambda node: ''.join('{{{}}}'.format(':'.join(map(str, n))) for n in node)
    for node in nodes:
        vals = [pid.get_partial(node) for pid in pids]
        vals = [0.0 if np.isclose(0, val, atol=1e-5, rtol=1e-5) else val for val in vals]
        table.add_row([stringify(node)] + vals)
    print(table.get_string())
