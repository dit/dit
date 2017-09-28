"""
The I_measure, defining an auxiliary variable to capture the redundancy between sources
"""

from __future__ import division

from .pid import BaseBivariatePID

from ..multivariate import coinformation
from ..utils import partitions
from ..distconst import RVFunctions, insert_rvf


def i_aux(d, inputs, output):
    """
    I_aux is maximum coinformation between all sources, targets, and aan arbitrary function of the sources

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_mmi for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    i_aux : float
        The value of I_aux.
    """
    d = d.coalesce(inputs + (output,))

    #parts = partitions(d.marginal(sum(inputs, ())).outcomes)
    parts = partitions(d.outcomes)
    bf = RVFunctions(d)
    extended_dists = [insert_rvf(d, bf.from_partition(part)) for part in parts]
    return max([coinformation(extended_dist) for extended_dist in extended_dists])


class PID_aux(BaseBivariatePID):
    """
    The maximum coinformation auxiliary random variable method
    """
    _name = "I_aux"
    _measure = staticmethod(i_aux)
