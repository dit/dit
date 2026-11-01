"""
Helpers shared by the partial information decompositions: labels of lattice nodes, and
the restricted and conditional distributions of groups of variables.
"""

from collections import defaultdict
from itertools import chain

import numpy as np

from ..distribution import Distribution

__all__ = ()


def _label(node):
    """
    Label an antichain as in `BasePID.to_string`, e.g. '{0}{1:2}'.
    """
    return "".join("{" + ":".join(map(str, element)) + "}" for element in node)


def _canonical(elements):
    """
    The lattice node of an antichain: minimal elements, ordered as in the lattice.
    """
    elements = {tuple(sorted(element)) for element in elements}
    minimal = [e for e in elements if not any(set(o) < set(e) for o in elements)]
    return tuple(sorted(minimal, key=lambda e: (len(e), e)))


def _positions(node, sources):
    """
    The antichain `node` as a set of groups of positions in `sources`.
    """
    return frozenset(frozenset(j for j, s in enumerate(sources) if set(s) <= set(e)) for e in node)


def _node(groups, sources, extra=()):
    """
    The lattice node whose elements join the sources at each group of positions, and `extra`.
    """
    return _canonical(tuple(chain(*(sources[j] for j in group), extra)) for group in groups)


def _indices(dist, rvs):
    """
    The positions in the outcomes of `dist` of the random variables `rvs` (indices or names).
    """
    names = list(dist.dims)
    return tuple(names.index(name) for name in dist._resolve_rv_names(list(rvs)))


def _support(dist):
    """
    The (outcome, probability) pairs of the support of `dist`.
    """
    return [(outcome, p) for outcome, p in zip(dist.outcomes, dist.pmf, strict=True) if p > 0]


def _from_table(table):
    """
    A Distribution from a mapping of outcome tuples to probabilities.
    """
    return Distribution(list(table), list(table.values()))


def _layout(groups):
    """
    Consecutive variable indices for groups of the given sizes.
    """
    starts = np.cumsum([0] + [len(group) for group in groups])
    return [tuple(range(a, b)) for a, b in zip(starts[:-1], starts[1:], strict=True)]


def _restricted(dist, groups):
    """
    The joint distribution of the variable groups, placed consecutively.
    """
    columns = list(chain(*groups))
    table = defaultdict(float)
    for outcome, p in _support(dist):
        table[tuple(outcome[c] for c in columns)] += p
    return _from_table(table), _layout(groups)


def _conditionals(dist, given, groups):
    """
    The distributions of the variable groups given each value of `given`.

    Returns
    -------
    conditionals : list of (tuple, float, Distribution)
        Each value of `given`, its probability, and the conditional distribution, with the
        groups placed consecutively.
    layout : list of tuples
        The indices of each group in the conditional distributions.
    """
    columns = list(chain(*groups))
    tables = defaultdict(lambda: defaultdict(float))
    for outcome, p in _support(dist):
        tables[tuple(outcome[c] for c in given)][tuple(outcome[c] for c in columns)] += p
    conditionals = []
    for value, table in tables.items():
        pz = sum(table.values())
        conditionals.append((value, pz, _from_table({outcome: p / pz for outcome, p in table.items()})))
    return conditionals, _layout(groups)
