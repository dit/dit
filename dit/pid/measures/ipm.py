"""
I_plus/minus, by Finn & Lizier
"""

import numpy as np

from ..pid import BasePID


class PID_PM(BasePID):
    """
    The Finn & Lizier partial information decomposition.
    """

    _name = "I_±"

    @staticmethod
    def _measure(dist, sources, target):
        """
        """
        dist = dist.coalesce(sources + (target,))
        source_dists = [dist.marginal([i]) for i, _ in enumerate(sources)]
        source_target_dists = [dist.marginal([i, len(sources)]) for i, _ in enumerate(sources)]
        p_s_g_ts = [d.condition_on([1]) for d in source_target_dists]

        def min_h_s(outcome):
            return min(-np.log2(source_dists[i][(e,)]) for i, e in enumerate(outcome[:-1]))

        def min_h_s_g_t(outcome):
            t = outcome[-1]
            indexes = [p_s_g_t[0].outcomes.index((t,)) for p_s_g_t in p_s_g_ts]
            return min(-np.log2(p_s_g_ts[i][1][j][(e,)]) for i, (e, j) in enumerate(zip(outcome[:-1], indexes)))

        r_plus = np.nansum([dist[outcome] * min_h_s(outcome) for outcome in dist.outcomes])

        r_minus = np.nansum([dist[outcome] * min_h_s_g_t(outcome) for outcome in dist.outcomes])

        return r_plus - r_minus
