"""
I^sx, Shared Exclusion Partial Information Decomposition by
Makkeh, A. & Gutknecht, A. & Wibral, M. (2021). Introducing a differentiable measure of pointwise shared information.
Phys. Rev. E 103, 032149
"""

import numpy as np

from ..pid import BasePointwisePID

__all__ = ("PID_SX",)


def _prob_union(dist, outcome, var_sets):
    """Probability of the union of events where at least one var_set matches outcome."""
    mask = np.zeros(len(dist.outcomes), dtype=bool)
    for var_set in var_sets:
        mask |= np.array(
            [all(dist.outcomes[rlz][i] == outcome[i] for i in var_set) for rlz in range(len(dist.outcomes))]
        )
    return np.sum(dist.pmf, where=mask)


class PID_SX(BasePointwisePID):
    """
    Shared Exclusion Partial Information Decomposition (I^sx) as described in:

    Makkeh, A. & Gutknecht, A. & Wibral, M. (2021). Introducing a differentiable measure of pointwise shared information.
    Phys. Rev. E 103, 032149

    When constructed with ``pointwise=True``, also computes the per-outcome
    pointwise PID (PPID) and the informative / misinformative decomposition
    (i^sx+ and i^sx-).
    """

    _name = "I_sx"

    @staticmethod
    def _pointwise_measure(dist, sources, target, **kwargs):
        """
        Per joint-outcome pointwise shared information:
        i^sx(t : sources) = i^sx+(s) - i^sx-(s, t)

        Returns
        -------
        pw : dict
            ``{outcome: float}`` for every joint outcome in *dist*.
        """
        dist_marg = dist.marginalize(target)
        dist_target = dist.marginal(target)

        pw = {}
        for outcome in dist.outcomes:
            plus = -np.log2(_prob_union(dist_marg, outcome, sources))
            minus = -np.log2(
                _prob_union(dist, outcome, tuple(s + target for s in sources))
                / dist_target[tuple(outcome[t] for t in target)]
            )
            pw[outcome] = plus - minus

        return pw

    @staticmethod
    def _pointwise_measure_parts(dist, sources, target, **kwargs):
        """
        Informative (+) and misinformative (-) components of the pointwise
        shared information:

        * i^sx+(s) = -log2 P(union of source events)
        * i^sx-(s,t) = -log2 P(union of (source,target) events) / P(t)

        Returns
        -------
        plus_dict, minus_dict : dict, dict
        """
        dist_marg = dist.marginalize(target)
        dist_target = dist.marginal(target)

        plus_dict = {}
        minus_dict = {}
        for outcome in dist.outcomes:
            plus_dict[outcome] = -np.log2(_prob_union(dist_marg, outcome, sources))
            minus_dict[outcome] = -np.log2(
                _prob_union(dist, outcome, tuple(s + target for s in sources))
                / dist_target[tuple(outcome[t] for t in target)]
            )

        return plus_dict, minus_dict
