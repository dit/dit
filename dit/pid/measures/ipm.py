"""
I_plus/minus, by Finn & Lizier (2018).

Finn, C. & Lizier, J.T. (2018). Pointwise Partial Information Decomposition
Using the Specificity and Ambiguity Lattices. Entropy, 20(4), 297.
"""

import numpy as np

from ..pid import BasePointwisePID

__all__ = ("PID_PM",)


class PID_PM(BasePointwisePID):
    """
    The Finn & Lizier partial information decomposition (I_±).

    The measure decomposes into:
    * h_s (specificity / plus): min over sources of the source surprisal
    * h_s|t (ambiguity / minus): min over sources of the conditional surprisal

    When constructed with ``pointwise=True``, also computes the per-outcome
    pointwise PID and the specificity / ambiguity decomposition.
    """

    _name = "I_±"

    @classmethod
    def _pointwise_measure(cls, dist, sources, target):
        """
        Per-outcome pointwise redundancy:
        i_±(outcome) = min_h_s(outcome) - min_h_s|t(outcome)

        Returns a dict keyed by the original distribution's outcomes.
        """
        plus, minus = cls._pointwise_measure_parts(dist, sources, target)
        return {outcome: plus[outcome] - minus[outcome] for outcome in plus}

    @staticmethod
    def _pointwise_measure_parts(dist, sources, target):
        """
        Specificity (+) and ambiguity (-) components:
        * plus: min_h_s(outcome) = min over sources of -log2 P(s_i)
        * minus: min_h_s|t(outcome) = min over sources of -log2 P(s_i | t)

        Returns (plus_dict, minus_dict) keyed by original outcomes.
        """
        dist = dist.copy(base="linear") if dist.is_log() else dist
        # `marginal` orders variables by index, so build lookup keys the same way.
        source_idx = [sorted(s) for s in sources]
        target_idx = sorted(target)
        joint_idx = [sorted(list(s) + list(target)) for s in sources]
        source_marginals = [dist.marginal(idx) for idx in source_idx]
        target_marginal = dist.marginal(target_idx)
        source_target_joints = [dist.marginal(idx) for idx in joint_idx]

        def key(outcome, idx):
            return tuple(outcome[i] for i in idx)

        plus_dict = {}
        minus_dict = {}
        for outcome in dist.outcomes:
            plus_dict[outcome] = min(
                -np.log2(marg[key(outcome, idx)]) for marg, idx in zip(source_marginals, source_idx, strict=True)
            )

            p_t = target_marginal[key(outcome, target_idx)]
            h_s_g_t_vals = []
            for joint, idx in zip(source_target_joints, joint_idx, strict=True):
                p_s_g_t = joint[key(outcome, idx)] / p_t if p_t > 0 else 0.0
                h_s_g_t_vals.append(-np.log2(p_s_g_t))
            minus_dict[outcome] = min(h_s_g_t_vals)

        return plus_dict, minus_dict
