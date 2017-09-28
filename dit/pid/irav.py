"""
The I_rav measure, defining a 'redundancy' auxiliary variable to capture the redundancy information between sources
"""

from __future__ import division

from .pid import BaseBivariatePID

from ..multivariate import coinformation
from ..utils import partitions
from ..distconst import RVFunctions, insert_rvf


def i_rav(d, inputs, output):
    """
    I_rav is maximum coinformation between all sources, targets, and aan arbitrary function of the sources

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_rav for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    i_rav : float
        The value of I_rav.
    """
    d = d.coalesce(inputs + (output,))

    input_parts = partitions(d.marginal(sum(inputs, ())).outcomes)
    #parts = partitions(d.outcomes)
    parts = ()
    for ipart in input_parts:
        #
    bf = RVFunctions(d)
    extended_dists = [insert_rvf(d, bf.from_partition(part)) for part in parts]
    return max([coinformation(extended_dist) for extended_dist in extended_dists])


class PID_rav(BaseBivariatePID):
    """
    The maximum coinformation auxiliary random variable method
    """
    _name = "I_rav"
    _measure = staticmethod(i_rav)
