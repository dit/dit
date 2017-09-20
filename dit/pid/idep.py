"""
The dependency-decomposition based unique measure partial information decomposition.
"""

from __future__ import division

import networkx as nx

from .pid import BaseUniquePID

from ..multivariate import coinformation
from ..profiles import DependencyDecomposition


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
        The value of I_dep for each individual input.
    """
    uniques = {}
    measure = {'I': lambda d: coinformation(d, [[0, 1], [2]])}
    if len(inputs) == 2:
        dm = d.coalesce(inputs + (output,))
        dd = DependencyDecomposition(dm, measures=measure, maxiters=maxiters)
        u_0 = min(dd.delta(edge, 'I') for edge in dd.edges(((0, 2),)))
        u_1 = min(dd.delta(edge, 'I') for edge in dd.edges(((1, 2),)))
        uniques[inputs[0]] = u_0
        uniques[inputs[1]] = u_1
    else:
        for input_ in inputs:
            others = sum([i for i in inputs if i != input_], ())
            dm = d.coalesce([input_, others, output])
            dd = DependencyDecomposition(dm, measures=measure, maxiters=maxiters)
            u = min(dd.delta(edge, 'I') for edge in dd.edges(((0, 2),)))
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
    measure = {'I': lambda d: coinformation(d, [invars, outvar])}
    dd = DependencyDecomposition(d, list(var_to_index.values()), measures=measure)
    uniques = {}
    for input_ in inputs:
        constraint = ((var_to_index[input_], var_to_index[output]),)
        u = min(dd.delta(edge, 'I') for edge in dd.edges(constraint))
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
    output_index = var_to_index[output]
    d = d.coalesce(list(sorted(var_to_index.keys(), key=lambda k: var_to_index[k])))
    invars = [ var_to_index[var] for var in inputs ]
    outvar = [ var_to_index[(var,)] for var in output ]
    measure = {'I': lambda d: coinformation(d, [invars, outvar])}
    dd = DependencyDecomposition(d, list(var_to_index.values()), measures=measure)
    uniques = {}
    for input_ in inputs:
        constraint = ((var_to_index[input_], output_index),)
        broja_style = lambda edge: all({output_index} < set(_) for _ in edge[0] if len(_) > 1)
        edge_set = (edge for edge in dd.edges(constraint) if broja_style(edge))
        u = min(dd.delta(edge, 'I') for edge in edge_set)
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
    measure = {'I': lambda d: coinformation(d, [invars, outvar])}
    dd = DependencyDecomposition(d, list(var_to_index.values()), measures=measure)
    uniques = {}
    for input_ in inputs:
        constraint = ((var_to_index[input_], var_to_index[output]),)
        edge_set = (edge for edge in dd.edges(constraint) if tuple(invars) in edge[0])
        u = min(dd.delta(edge, 'I') for edge in edge_set)
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
