"""
The I_mmi measure, briefly looked at by the BROJA team.
"""

from ...multivariate import coinformation
from ..pid import BasePID

__all__ = (
    'PID_MMI',
)


class PID_MMI(BasePID):
    """
    The minimum mutual information partial information decomposition.

    This measure is known to be accurate for gaussian random variables. It was also
    briefly and tangentially studied by the BROJA team.
    """

    _name = "I_mmi"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_mmi is the minimum mutual information between any source and the target.

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
        immi : float
            The value of I_mmi.
        """
        return min(coinformation(d, [source, target]) for source in sources)
