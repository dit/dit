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

    @staticmethod
    def _pointwise_measure(dist, sources, target, **kwargs):
        """
        Per-outcome pointwise redundancy:
        i_±(outcome) = min_h_s(outcome) - min_h_s|t(outcome)

        Returns a dict keyed by the original distribution's outcomes.
        """
        source_marginals = [dist.marginal(list(s)) for s in sources]
        target_marginal = dist.marginal(list(target))
        source_target_joints = [dist.marginal(list(s) + list(target)) for s in sources]

        pw = {}
        for outcome in dist.outcomes:
            source_vals = [tuple(outcome[i] for i in s) for s in sources]
            target_val = tuple(outcome[i] for i in target)

            min_h_s = min(-np.log2(source_marginals[i][sv]) for i, sv in enumerate(source_vals))

            p_t = target_marginal[target_val]
            h_s_g_t_vals = []
            for i, sv in enumerate(source_vals):
                p_st = source_target_joints[i][sv + target_val]
                p_s_g_t = p_st / p_t if p_t > 0 else 0.0
                h_s_g_t_vals.append(-np.log2(p_s_g_t))
            min_h_s_g_t = min(h_s_g_t_vals)

            pw[outcome] = min_h_s - min_h_s_g_t

        return pw

    @staticmethod
    def _pointwise_measure_parts(dist, sources, target, **kwargs):
        """
        Specificity (+) and ambiguity (-) components:
        * plus: min_h_s(outcome) = min over sources of -log2 P(s_i)
        * minus: min_h_s|t(outcome) = min over sources of -log2 P(s_i | t)

        Returns (plus_dict, minus_dict) keyed by original outcomes.
        """
        source_marginals = [dist.marginal(list(s)) for s in sources]
        target_marginal = dist.marginal(list(target))
        source_target_joints = [dist.marginal(list(s) + list(target)) for s in sources]

        plus_dict = {}
        minus_dict = {}
        for outcome in dist.outcomes:
            source_vals = [tuple(outcome[i] for i in s) for s in sources]
            target_val = tuple(outcome[i] for i in target)

            plus_dict[outcome] = min(-np.log2(source_marginals[i][sv]) for i, sv in enumerate(source_vals))

            p_t = target_marginal[target_val]
            h_s_g_t_vals = []
            for i, sv in enumerate(source_vals):
                p_st = source_target_joints[i][sv + target_val]
                p_s_g_t = p_st / p_t if p_t > 0 else 0.0
                h_s_g_t_vals.append(-np.log2(p_s_g_t))
            minus_dict[outcome] = min(h_s_g_t_vals)

        return plus_dict, minus_dict

    @classmethod
    def _measure(cls, dist, sources, target, **kwargs):
        """
        Averaged redundancy over the joint distribution.
        """
        pw = cls._pointwise_measure(dist, sources, target, **kwargs)
        return np.nansum([dist[outcome] * val for outcome, val in pw.items()])
