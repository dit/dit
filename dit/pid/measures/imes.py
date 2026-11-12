"""
A PID inspired by BROJA's * assumption
"""

from ...algorithms.pid_references import conditionally_independent
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_MES",)


class PID_MES(BaseBivariatePID):
    """
    The maximum entropy PID inspired by BROJA's * assumption.

    Redundancy is the coinformation of the sources and the target under the maximum
    entropy distribution with the same source-target marginals,
    q = p(t) prod_i p(x_i | t). The sources are independent given the target under q, so
    for two sources this is I_q(X_0 : X_1), which is also the redundancy of
    :class:`~dit.pid.PID_Do`.

    The distribution itself is not among the reference distributions, so the synergy, the
    redundancy minus the co-information of p, can be negative :cite:`vansanten2026between`.
    For example, two identical sources and a target copying them through a binary
    symmetric channel with crossover 0.1 give redundancy 0.320 and synergy -0.211 bits.
    """

    _name = "I_ME*"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_ME*, the coinformation under the maximum entropy distribution satisfying
        the * assumption of BROJA.

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
        return coinformation(conditionally_independent(d, sources, target))
