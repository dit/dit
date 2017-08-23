"""
The I_mmi measure, briefly looked at by the BROJA team.
"""

from __future__ import division

from .pid import BasePID

from ..multivariate import coinformation


def i_mmi(d, inputs, output):
    """
    I_mmi is the minimum mutual information between any input and the output.

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
    immi : float
        The value of I_mmi.
    """
    return min(coinformation(d, [input_, output]) for input_ in inputs)


class PID_MMI(BasePID):
    """
    The minimum mutual information partial information decomposition.

    This measure is known to be accurate for gaussian random variables. It was also
    briefly and tangentially studied by the BROJA team.
    """
    _name = "I_mmi"
    _measure = staticmethod(i_mmi)
