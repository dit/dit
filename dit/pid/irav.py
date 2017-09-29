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
    I_RAV is maximum coinformation between all sources, targets, and aan arbitrary function of the sources

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
    #print(f"input_outcome[1]: {outcomes[1][:-1]}")

    parts = ()
    #extended_dists = []
    for input_part in input_parts:
        #print(f"input part: {input_part}")
        part = ()
        for input_part_element in input_part:
            #print(input_part_element)
            part_element = ()
            for input_outcome in input_part_element:
                part_element += tuple(outcome for outcome in outcomes if outcome[:-1] == input_outcome)
            #print(f"part element: {part_element}")
            part += (part_element,)
        parts += (part,)
        #extended_dists += [insert_rvf(d, bf.from_partition(part))]
        #print(f"parts: {parts}")

    bf = RVFunctions(d)
    extended_dists = [insert_rvf(d, bf.from_partition(part)) for part in parts]
    return max([coinformation(extended_dist) for extended_dist in extended_dists])


class PID_RAV(BaseBivariatePID):
    """
    The maximum coinformation auxiliary random variable method
    """
    _name = "I_RAV"
    _measure = staticmethod(i_rav)
