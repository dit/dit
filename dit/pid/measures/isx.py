"""
I^sx, Shared Exclusion Partial Information Decomposition by
Makkeh, A. & Gutknecht, A. & Wibral, M. (2021). Introducing a differentiable measure of pointwise shared information.
Phys. Rev. E 103, 032149
"""

import numpy as np

from ..pid import BasePID

__all__ = (
    'PID_SX',
)


class PID_SX(BasePID):
    """
    Shared Exclusion Partial Information Decomposition (I^sx) as described in:

    Makkeh, A. & Gutknecht, A. & Wibral, M. (2021). Introducing a differentiable measure of pointwise shared information.
    Phys. Rev. E 103, 032149
    """

    _name = "I_sx"

    @staticmethod
    def _measure(dist, sources, target):
        """
        """

        dist_marg = dist.marginalize(target)
        dist_target = dist.marginal(target)

        def prob_union(dist, outcome, var_sets):
            # sum over all outcomes where one of the sets of sources in "sources" is equal to the corresponding elements in outcome
            mask = np.zeros(len(dist.outcomes), dtype=bool)
            for var_set in var_sets:
                mask |= np.array([all(dist.outcomes[rlz][i] == outcome[i] for i in var_set) for rlz in range(len(dist.outcomes))])
            return np.sum(dist.pmf, where=mask)

        def i_sx_plus(outcome):
            return -np.log2(prob_union(dist_marg, outcome, sources))

        I_sx_plus = np.nansum([dist_marg[outcome] * i_sx_plus(outcome) for outcome in dist_marg.outcomes])

        def i_sx_minus(outcome):
            return -np.log2(prob_union(dist, outcome, tuple(s + target for s in sources)) / dist_target[tuple(outcome[t] for t in target)])

        I_sx_minus = np.nansum([dist[outcome] * i_sx_minus(outcome) for outcome in dist.outcomes])

        return I_sx_plus - I_sx_minus
