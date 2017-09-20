"""
I_plus/minus, by Finn & Lizier
"""

from __future__ import division

import numpy as np

from .pid import BasePID


def i_pm(dist, inputs, output):
    """
    """
    dist = dist.coalesce(inputs + (output,))
    input_dists = [dist.marginal([i]) for i, _ in enumerate(inputs)]
    inout_dists = [dist.marginal([i, len(inputs)]) for i, _ in enumerate(inputs)]
    p_s_g_ts = [d.condition_on([1]) for d in inout_dists]

    def min_h_s(outcome):
        return min(-np.log2(input_dists[i][(e,)]) for i, e in enumerate(outcome[:-1]))

    def min_h_s_g_t(outcome):
        t = outcome[-1]
        indexes = [p_s_g_t[0].outcomes.index((t,)) for p_s_g_t in p_s_g_ts]
        return min(-np.log2(p_s_g_ts[i][1][j][(e,)]) for i, (e, j) in enumerate(zip(outcome[:-1], indexes)))

    r_plus = sum(dist[outcome]*min_h_s(outcome) for outcome in dist.outcomes)

    r_minus = sum(dist[outcome]*min_h_s_g_t(outcome) for outcome in dist.outcomes)

    return r_plus - r_minus


class PID_PM(BasePID):
    """
    The Finn & Lizier partial information decomposition.
    """
    _name = "I_pm"
    _measure = staticmethod(i_pm)
