"""
The I_wedge measure, as proposed by Griffith et al.
"""

from __future__ import division

from .pid import BasePID

from .. import Distribution
from ..algorithms import insert_meet
from ..multivariate import coinformation


def i_wedge(d, inputs, output):
    """
    Compute I_wedge(inputs : output) = I(meet(inputs) : output)

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_wedge for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    iwedge : float
        The value of I_wedge.
    """
    d = d.coalesce(inputs+(output,))
    d = Distribution(d.outcomes, d.pmf, sample_space=d.outcomes)
    d = insert_meet(d, -1, d.rvs[:-1])
    return coinformation(d, [d.rvs[-2], d.rvs[-1]])


class PID_GK(BasePID):
    """
    The Griffith et al partial information decomposition.

    This PID is known to produce negative partial information values.
    """
    _name = "I_GK"
    _measure = staticmethod(i_wedge)
