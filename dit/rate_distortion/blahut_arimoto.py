"""
"""

from __future__ import division

import numpy as np

from .distortions import hamming_distortion
from .rate_distortion import RateDistortionResult
from ..divergences.pmf import (earth_movers_distance,
                               maximum_correlation,
                               relative_entropy,
                               variational_distance,
                               )


###############################################################################
# Rate-Distortion

def _blahut_arimoto(p_x, beta, q_y_x, distortion, max_iters=100):
    """
    """
    def q_xy(q_y_x):
        q_xy = p_x[:, np.newaxis] * q_y_x
        return q_xy

    def next_q_y(q_y_x):
        q_y = np.matmul(p_x, q_y_x)
        return q_y

    def next_q_y_x(q_y, q_y_x):
        d = distortion(p_x, q_y_x)
        q_y_x = q_y * np.exp2(-beta * d)
        q_y_x /= q_y_x.sum(axis=1, keepdims=True)
        return q_y_x

    def av_dist(q_y_x, dist):
        d = np.matmul(p_x, (q_y_x * dist)).sum()
        return d

    def next_rd(q_y, q_y_x):
        q_y = next_q_y(q_y_x)
        q_y_x = next_q_y_x(q_y, q_y_x)
        d = av_dist(q_y_x, distortion(p_x, q_y_x))
        return q_y, q_y_x, d

    q_y = next_q_y(q_y_x)
    prev_d = 0
    d = av_dist(q_y_x, distortion(p_x, q_y_x))

    iters = 0
    while not np.isclose(prev_d, d) and iters < max_iters:
        iters += 1
        (q_y, q_y_x, d), prev_d = next_rd(q_y, q_y_x), d

    q = q_xy(q_y_x)
    r = np.nansum(q * np.log2(q / (q.sum(axis=0, keepdims=True) * q.sum(axis=1, keepdims=True))))
    result = RateDistortionResult(r, d)

    return result, q


def blahut_arimoto(p_x, beta, distortion=hamming_distortion, max_iters=100, restarts=100):
    """
    Todo
    ----
    latin hypercube sampling
    """
    n = len(p_x)
    candidates = []
    for i in range(restarts):
        if i == 0:
            q_y_x = np.ones((n, n)) / n
        elif i == 1:
            q_y_x = np.zeros((n, n))
            q_y_x[0, :] = 1
        else:
            q_y_x = np.random.random((n, n))
            q_y_x /= q_y_x.sum(axis=1, keepdims=True)
        result = _blahut_arimoto(p_x=p_x,
                                 beta=beta,
                                 q_y_x=q_y_x,
                                 distortion=distortion,
                                 max_iters=max_iters
                                 )
        candidates.append(result)

    rd = min(candidates, key=lambda result: result[0].rate + beta*result[0].distortion)
    return rd


###############################################################################
# Information Bottleneck

def blahut_arimoto_ib(p_xy, beta, divergence=relative_entropy, max_iters=100, restarts=100):
    """
    """
    p_x = p_xy.sum(axis=1)
    p_y_x = p_xy / p_xy.sum(axis=1, keepdims=True)

    def next_q_y_t(q_t_x):
        q_xyt = q_t_x[:, np.newaxis, :] * p_xy[:, :, np.newaxis]
        q_yt = q_xyt.sum(axis=0)
        q_y_t = q_yt / q_yt.sum(axis=0, keepdims=True)
        q_y_t[np.isnan(q_y_t)] = 0
        return q_y_t

    def distortion(p_x, q_t_x):
        """
        """
        q_y_t = next_q_y_t(q_t_x)
        distortions = np.asarray([divergence(a, b) for b in q_y_t.T for a in p_y_x]).reshape(q_y_t.shape)
        return distortions

    return blahut_arimoto(p_x=p_x,
                          beta=beta,
                          distortion=distortion,
                          max_iters=max_iters,
                          restarts=restarts
                          )
