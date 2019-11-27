"""
The I_RR measure, described in ``Temporal information partitioning:
Characterizing synergy, uniqueness, and redundancy in interacting environmental
variables'' by Goodwell & Kumar.
"""

from __future__ import division

from ..pid import BaseBivariatePID

from ...multivariate import coinformation, entropy


class PID_RR(BaseBivariatePID):
    """
    The minimum mutual information partial information decomposition.

    This measure is known to be accurate for gaussian random variables. It was also
    briefly and tangentially studied by the BROJA team.
    """
    _name = "I_rr"

    @staticmethod
    def _measure(d, inputs, output):
        """
        I_rr, the rescaled redundancy.

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
        irr : float
            The value of I_rr.
        """
        R_min = max([0, coinformation(d, inputs + (output,))])
        R_mmi = min(coinformation(d, [input_, output]) for input_ in inputs)
        I_s = coinformation(d, inputs)/min([entropy(d, input_) for input_ in inputs])
        return R_min + I_s * (R_mmi - R_min)
