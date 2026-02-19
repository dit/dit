"""
A PID inspired by BROJA's * assumption
"""

from dit.algorithms import maxent_dist
from dit.multivariate import coinformation as I
from dit.pid.pid import BaseBivariatePID

__all__ = (
    'PID_MES',
)


class PID_MES(BaseBivariatePID):
    """
    """

    _name = "I_ME*"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_ME*, the maximum entropy distribution satisfying the * assumption of
        BROJA.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_mes for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_mes : float
            The value of I_ME*.
        """
        dp = maxent_dist(d, [source + target for source in sources])
        i_mes = I(dp)
        return i_mes
