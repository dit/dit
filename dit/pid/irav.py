"""
The I_rav measure, defining a 'redundancy' auxiliary variable to capture the redundancy information between sources
"""

from __future__ import division

from .pid import BasePID

from ..multivariate import coinformation
from ..utils import partitions, extended_partition
from ..distconst import RVFunctions, insert_rvf


def i_rav(d, inputs, output):
    """
    I_RAV is maximum coinformation between all sources, targets, and an 
    arbitrary function of the sources.

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
        The value of I_RAV.
    """
    d = d.coalesce(inputs + (output,))

    input_parts = partitions(d.marginal(sum(d.rvs[:-1], [])).outcomes)
    outcomes = d.outcomes
    ctor = d._outcome_ctor
    idxs = list(range(len(inputs)))

    parts = [extended_partition(outcomes, idxs, input_part, ctor) for input_part in input_parts]

    bf = RVFunctions(d)
    extended_dists = [insert_rvf(d, bf.from_partition(part)) for part in parts]
    return max([coinformation(extended_dist) for extended_dist in extended_dists])


class PID_RAV(BasePID):
    """
    The maximum coinformation auxiliary random variable method
    """
    _name = "I_RAV"
    _measure = staticmethod(i_rav)
