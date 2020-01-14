"""
The I_RR measure, described in ``Temporal information partitioning:
Characterizing synergy, uniqueness, and redundancy in interacting environmental
variables'' by Goodwell & Kumar.
"""

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
    def _measure(d, sources, target):
        """
        I_rr, the rescaled redundancy.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_mmi for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        irr : float
            The value of I_rr.
        """
        R_min = max([0, coinformation(d, sources + (target,))])
        R_mmi = min(coinformation(d, [source, target]) for source in sources)
        I_s = coinformation(d, sources)/min([entropy(d, source) for source in sources])
        return R_min + I_s * (R_mmi - R_min)
