"""
Lattice utilities for the partial information decomposition.
"""

from itertools import combinations

import networkx as nx

from ..utils import powerset

__all__ = ['pid_lattice',
           'sort_key',
           'ascendants',
           'descendants',
           ]

def comparable(a, b):
    """
    Tests if two sets of comparable.

    Parameters
    ----------
    a : set
        One set.
    b : set
        The other set.

    Returns
    -------
    comp : bool
        True if `a` is a subset of `b` or vice versa.
    """
    return a < b or b < a


def antichain(ss):
    """
    Tests if a set of sets forms an antichain.

    Parameters
    ----------
    ss : set of sets
        The potential antichain.

    Returns
    -------
    ac : bool
        True if the sets in `ss` are pairwise non-comparable.
    """
    if not ss:
        return False
    return all(not comparable(frozenset(a), frozenset(b)) for a, b in combinations(ss, 2))


def parent(a, b):
    """
    Tests if `a` is a parent of `b` in the Williams & Beer lattice of antichains.

    Parameters
    ----------
    a : iterable of iterables
        One antichain.
    b : iterable of iterables
        Another antichain.

    Returns
    -------
    parent : bool
        True if, for each set in `b`, there exists a set in `a` which is a subset of that set.
    """
    return all(any(frozenset(aa) <= frozenset(bb) for aa in a) for bb in b)


def pid_lattice(variables):
    """
    Construct the Williams & Beer lattice of antichains.

    Parameters
    ----------
    n : int
        The number of input variables.

    Returns
    -------
    lattice : nx.DiGraph
        The lattice of antichains.
    """
    combos = (sum(s, tuple()) for s in powerset(variables))
    next(combos)  # remove null
    nodes = [ss for ss in powerset(combos) if antichain(ss)]

    lattice = nx.DiGraph()
    lattice.add_nodes_from(nodes)

    for a, b in combinations(nodes, 2):
        if parent(a, b):
            lattice.add_edge(b, a)
        elif parent(b, a):
            lattice.add_edge(a, b)

    for a, b in list(lattice.edges()):
        if any(n > 2 for n in map(len, nx.all_simple_paths(lattice, a, b))):
            lattice.remove_edge(a, b)

    lattice.root = next(iter(nx.topological_sort(lattice)))

    return lattice


def sort_key(lattice):
    """
    A key for sorting the nodes of a PID lattice.

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice to sort.

    Returns
    -------
    key : function
        A function on nodes which returns the properties from which the lattice should be ordered.
    """
    pls = nx.shortest_path_length(lattice, source=lattice.root)

    def key(node):
        depth = pls[node]
        size = len(node)
        return depth, -size, node

    return key


def ascendants(lattice, node, self=False):
    """
    Returns the nodes greater than `node`.

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice.
    node :
        The node in the lattice.

    Returns
    -------
    nodes : list
        A list of nodes greater than `node` in the lattice.
    """
    nodes = list(nx.bfs_tree(lattice.reverse(), node))
    if not self:
        nodes.remove(node)
    return nodes


def descendants(lattice, node, self=False):
    """
    Returns the nodes less than `node`.

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice.
    node :
        The node in the lattice.

    Returns
    -------
    nodes : list
        A list of nodes less than `node` in the lattice.
    """
    nodes = list(nx.bfs_tree(lattice, node))
    if not self:
        nodes.remove(node)
    return nodes

def least_upper_bound(lattice, nodes, predicate=None):
    """
    Find the least upper bound of `nodes` satisfying `predicate`.

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice to work with.
    nodes : iterable
        The nodes to find the least upper bound of.
    predicate : func(node, nodes) -> bool, None
        A criteria

    Returns
    -------
    lub : node, None
        The least upper bound of `nodes` satisfying `predicate`. None if one doesn't exist.
    """
    parents = ascendants(lattice, nodes[0], self=True)
    snodes = set(nodes)
    for node in sorted(parents, key=sort_key(lattice)):
        if snodes <= set(descendants(lattice, node, self=True)):
            if predicate is not None:
                if predicate(node, nodes):
                    return node
            else:
                return node
