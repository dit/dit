"""
The I_mmi measure, briefly looked at by the BROJA team.
"""

from ...multivariate import coinformation
from ..pid import BasePID
from .imin import _specific_informations

__all__ = ("PID_MMI",)


class PID_MMI(BasePID):
    """
    The minimum mutual information partial information decomposition.

    For jointly Gaussian sources and a univariate target it coincides with every
    decomposition satisfying BROJA's assumption (*), such as BROJA and I_min
    :cite:`barrett2015exploration`. It is also the largest redundancy of any nonnegative
    decomposition, attaining the bound that monotonicity places on every node at once
    :cite:`vansanten2026between`, so its synergy is the largest and its unique
    informations the smallest of any nonnegative decomposition. It was also briefly
    and tangentially studied by the BROJA team.
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
        if d.is_symbolic():
            from ...symbolic import symbolic_min

            return symbolic_min([coinformation(d, [source, target]) for source in sources])
        mis = []
        for source in sources:
            p_t, specific = _specific_informations(d, source, target)
            mis.append(sum(p * specific[t] for t, p in p_t.items()))
        return float(min(mis))
