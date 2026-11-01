"""
The I_RR measure, described in ``Temporal information partitioning:
Characterizing synergy, uniqueness, and redundancy in interacting environmental
variables'' by Goodwell & Kumar.
"""

from ...multivariate import coinformation, entropy
from ..pid import BaseBivariatePID

__all__ = ("PID_RR",)


class PID_RR(BaseBivariatePID):
    """
    The rescaled redundancy of Goodwell & Kumar.

    Redundancy interpolates between max(0, coinformation) and the minimum mutual
    information, in proportion to the normalized dependence I(X_0 : X_1) / min_i H(X_i)
    between the sources.
    """

    _name = "I_rr"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_rr, the rescaled redundancy.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_rr for.
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
        h_min = min(entropy(d, source) for source in sources)
        I_s = coinformation(d, sources) / h_min if h_min > 0 else 0.0
        return R_min + I_s * (R_mmi - R_min)
