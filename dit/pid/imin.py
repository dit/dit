"""
The I_min measure as proposed by Williams & Beer.
"""

from __future__ import division

import numpy as np

from .pid import BasePID


def s_i(d, input_, output, output_value):
    """
    Compute the specific mutual information I(input_ : output=output_value)

    Parameters
    ----------
    d : Distribution
        The distribution from which this quantity is to be calculated.
    input_ : iterable
        The input aggregate variable.
    output : iterable
        The output aggregate variable.
    output_value : iterable
        The value of the output.

    Returns
    -------
    s : float
        The specific information
    """
    pp_s, pp_a_s = d.condition_on(output, rvs=input_)
    p_s = pp_s[output_value]
    p_a_s = pp_a_s[pp_s.outcomes.index(output_value)]
    pp_a, pp_s_a = d.condition_on(input_, rvs=output)
    p_s_a = {a: pp[output_value] for a, pp in zip(pp_a.outcomes, pp_s_a)}

    return np.nansum([p_a_s[a] * np.log2(psa / p_s) for a, psa in p_s_a.items()])


def i_min(d, inputs, output):
    """
    Compute I_min(inputs : output) =
        \sum_{s \in output} p(s) min_{input_ \in inputs} I(input_ : output=s)

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_min for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    imin : float
        The value of I_min.
    """
    p_s = d.marginal(output)
    return sum(p_s[s] * min(s_i(d, input_, output, s) for input_ in inputs) for s in p_s.outcomes)


class PID_WB(BasePID):
    """
    The Williams & Beer partial information decomposition.
    """
    _name = "I_min"
    _measure = staticmethod(i_min)
