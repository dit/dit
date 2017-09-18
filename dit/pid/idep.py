"""
The dependency-decomposition based unique measure partial information decomposition.
"""

import networkx as nx

from .pid import BaseUniquePID

from ..multivariate import coinformation
from ..profiles import DependencyDecomposition


def edges(dd, var):
    """
    Find all the edges within the dependency decomposition which are the restriction of `var`.

    Parameters
    ----------
    dd : DependencyDecomposition
        The dependency decomposition in question.
    var : iterable
        The variable which is added among the edges to be found.

    Returns
    -------
    edges : set
        The set of edges (a, b) where a restricts `var` but b does not.
    """
    edge_set = set()
    for a, b in dd._lattice.edges():
        if set(var) <= set(a) - set(b) and nx.has_path(dd._lattice, a, b):
            edge_set |= {(a, b)}
    return edge_set


def delta(dd, m, a, b):
    """
    Return the difference in measure `m` along the edge (`a`, `b`).

    Parameters
    ----------
    dd : DependencyDecomposition
        The dependency decomposition to compute the edge delta with.
    m : str
        The name of the measure to use.
    a : iterable
        The parent node.
    b : iterable
        The child node.

    Returns
    -------
    diff : float
        The difference in the measure at `a` and `b`.
    """
    return dd.atoms[a][m] - dd.atoms[b][m]


def i_dep(d, inputs, output, maxiters=1000):
    """
    This computes unique information as min(delta(I(inputs : output))) where delta
    is taken over the dependency decomposition.

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_dep for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    idep : dict
        The value of I_dep_a for each individual input.
    """
    uniques = {}
    if len(inputs) == 2:
        dm = d.coalesce(inputs + (output,))
        dd = DependencyDecomposition(dm, measures={'I': lambda d: coinformation(d, [inputs, output])}, maxiters=maxiters)
        edge_set_0 = (edge for edge in edges(dd, ((0, 2),)))
        edge_set_1 = (edge for edge in edges(dd, ((1, 2),)))
        uniques[inputs[0]] = min(delta(dd, 'I', *edge) for edge in edge_set_0)
        uniques[inputs[1]] = min(delta(dd, 'I', *edge) for edge in edge_set_1)
    else:
        for input_ in inputs:
            others = sum([i for i in inputs if i != input_], ())
            dm = d.coalesce([input_, others, output])

            dd = DependencyDecomposition(dm, measures={'I': lambda d: coinformation(d, [[0, 1], [2]])}, maxiters=maxiters)
            edge_set = (edge for edge in edges(dd, ((0, 2),)))
            u = min(delta(dd, 'I', *edge) for edge in edge_set)
            uniques[input_] = u

    return uniques


class PID_dep(BaseUniquePID):
    """
    The dependency partial information decomposition, as defined by James at al.
    """
    _name = "I_dep"
    _measure = staticmethod(i_dep)


# pragma: no cover
def i_dep_a(d, inputs, output):
    """
    This computes unique information as min(delta(I(inputs : output))) where delta
    is taken over the dependency decomposition.

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_dep for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    idepa : dict
        The value of I_dep_a for each individual input.
    """
    var_to_index = { var: i for i, var in enumerate(inputs+(output,)) }
    d = d.coalesce(list(sorted(var_to_index.keys(), key=lambda k: var_to_index[k])))
    invars = [ var_to_index[var] for var in inputs ]
    outvar = [ var_to_index[(var,)] for var in output ]
    dd = DependencyDecomposition(d, list(var_to_index.values()), measures={'I': lambda d: coinformation(d, [invars, outvar])})
    uniques = {}
    for input_ in inputs:
        edge_set = (edge for edge in edges(dd, ((var_to_index[input_], var_to_index[output]),)))
        u = min(delta(dd, 'I', *edge) for edge in edge_set)
        uniques[input_] = u
    return uniques


# pragma: no cover
class PID_dep_a(BaseUniquePID):
    """
    The dependency partial information decomposition, as defined by James at al.
    """
    _name = "I_dep_a"
    _measure = staticmethod(i_dep_a)


# pragma: no cover
def i_dep_b(d, inputs, output):
    """
    This computes unique information as min(delta(I(inputs : output))) where delta
    is taken over a restricted dependency decomposition which never constrains dependencies
    among the inputs.

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_rdep for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    idepb : dict
        The value of I_dep_b for each individual input.
    """
    var_to_index = { var: i for i, var in enumerate(inputs+(output,)) }
    d = d.coalesce(list(sorted(var_to_index.keys(), key=lambda k: var_to_index[k])))
    invars = [ var_to_index[var] for var in inputs ]
    outvar = [ var_to_index[(var,)] for var in output ]
    dd = DependencyDecomposition(d, list(var_to_index.values()), measures={'I': lambda d: coinformation(d, [invars, outvar])})
    uniques = {}
    for input_ in inputs:
        edge_set = (edge for edge in edges(dd, ((var_to_index[input_], var_to_index[output]),)) if all({var_to_index[output]} < set(_) for _ in edge[0] if len(_) > 1))
        u = min(delta(dd, 'I', *edge) for edge in edge_set)
        uniques[input_] = u
    return uniques


# pragma: no cover
class PID_dep_b(BaseUniquePID):
    """
    The reduced dependency partial information decomposition, as defined by James at al.

    Notes
    -----
    This decomposition is known to be inconsistent
    """
    _name = "I_dep_b"
    _measure = staticmethod(i_dep_b)


# pragma: no cover
def i_dep_c(d, inputs, output):
    """
    This computes unique information as min(delta(I(inputs : output))) where delta
    is taken over a restricted dependency decomposition which never constrains dependencies
    among the inputs.

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_rdep for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    idepc : dict
        The value of I_dep_c for each individual input.
    """
    var_to_index = { var: i for i, var in enumerate(inputs+(output,)) }
    d = d.coalesce(list(sorted(var_to_index.keys(), key=lambda k: var_to_index[k])))
    invars = [ var_to_index[var] for var in inputs ]
    outvar = [ var_to_index[(var,)] for var in output ]
    dd = DependencyDecomposition(d, list(var_to_index.values()), measures={'I': lambda d: coinformation(d, [invars, outvar])})
    uniques = {}
    for input_ in inputs:
        edge_set = (edge for edge in edges(dd, ((var_to_index[input_], var_to_index[output]),)) if tuple(invars) in edge[0])
        u = min(delta(dd, 'I', *edge) for edge in edge_set)
        uniques[input_] = u
    return uniques


# pragma: no cover
class PID_dep_c(BaseUniquePID):
    """
    The reduced dependency partial information decomposition, as defined by James at al.

    Notes
    -----
    This decomposition can result in subadditive redundancy.
    """
    _name = "I_dep_c"
    _measure = staticmethod(i_dep_c)
