"""
The Blahut-Arimoto algorithm for solving the rate-distortion problem.
"""

import numpy as np

from ..divergences.pmf import relative_entropy
from ..math.sampling import sample_simplex
from .distortions import hamming_distortion
from .rate_distortion import RateDistortionResult

__all__ = (
    'blahut_arimoto',
    'blahut_arimoto_ib',
)


###############################################################################
# Rate-Distortion

def _blahut_arimoto(p_x, beta, q_y_x, distortion, max_iters=100):
    """
    Perform the Blahut-Arimoto algorithm.

    Parameters
    ----------
    p_x : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    distortion : np.ndarray
        The distortion matrix.
    max_iters : int
        The maximum number of iterations.

    Returns
    -------
    result : RateDistortionResult
        A rate, distortion pair.
    q_xy : np.ndarray
        The joint distribution q(x, y).
    """
    def q_xy(q_y_x):
        """
        :math:`q(x,y) = q(y|x)p(x)`
        """
        q_xy = p_x[:, np.newaxis] * q_y_x
        return q_xy

    def next_q_y(q_y_x):
        """
        :math:`q(y) = \\sum_x q(y|x)p(x)`
        """
        q_y = np.matmul(p_x, q_y_x)
        return q_y

    def next_q_y_x(q_y, q_y_x):
        """
        :math:`q(y|x) = q(y) 2^{-\\beta * distortion}`
        """
        d = distortion(p_x, q_y_x)
        q_y_x = q_y * np.exp2(-beta * d)
        q_y_x /= q_y_x.sum(axis=1, keepdims=True)
        return q_y_x

    def av_dist(q_y_x, dist):
        """
        :math:`<dist> = \\sum_{x, t} q(x,t) * d(x,t)`
        """
        d = np.matmul(p_x, (q_y_x * dist)).sum()
        return d

    def next_rd(q_y, q_y_x):
        """
        Iterate the BA equations.
        """
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
    Perform a robust form of the Blahut-Arimoto algorithms.

    Parameters
    ----------
    p_x : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    distortion : np.ndarray
        The distortion matrix.
    max_iters : int
        The maximum number of iterations.
    restarts : int
        The number of initial conditions to try.

    Returns
    -------
    result : RateDistortionResult
        The rate, distortion pair.
    q_xy : np.ndarray
        The distribution p(x, y) which achieves the optimal rate, distortion.
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
            q_y_x = sample_simplex(n, n)

        result = _blahut_arimoto(p_x=p_x,
                                 beta=beta,
                                 q_y_x=q_y_x,
                                 distortion=distortion,
                                 max_iters=max_iters
                                 )
        candidates.append(result)

    rd = min(candidates, key=lambda result: result[0].rate + beta * result[0].distortion)
    return rd


###############################################################################
# Information Bottleneck

def blahut_arimoto_ib(p_xy, beta, divergence=relative_entropy, max_iters=100, restarts=250):  # pragma: no cover
    """
    Perform a robust form of the Blahut-Arimoto algorithms.

    Parameters
    ----------
    p_xy : np.ndarray
        The pmf to work with.
    beta : float
        The beta value for the optimization.
    q_y_x : np.ndarray
        The initial condition to work with.
    divergence : func
        The divergence measure to construct a distortion from: D(p(Y|x)||q(Y|t)).
    max_iters : int
        The maximum number of iterations.
    restarts : int
        The number of initial conditions to try.

    Returns
    -------
    result : RateDistortionResult
        The rate, distortion pair.
    q_xyt : np.ndarray
        The distribution p(x, y, t) which achieves the optimal rate, distortion.
    """
    p_x = p_xy.sum(axis=1)
    p_y_x = p_xy / p_xy.sum(axis=1, keepdims=True)

    def next_q_y_t(q_t_x):
        """
        :math:`q(y|t) = (\\sum_x p(x, y) * q(t|x)) / q(t)`
        """
        q_xyt = q_t_x[:, np.newaxis, :] * p_xy[:, :, np.newaxis]
        q_ty = q_xyt.sum(axis=0).T
        q_y_t = q_ty / q_ty.sum(axis=1, keepdims=True)
        q_y_t[np.isnan(q_y_t)] = 1
        return q_y_t

    def distortion(p_x, q_t_x):
        """
        :math:`d(x, t) = D[ p(Y|x) || q(Y|t) ]`
        """
        q_y_t = next_q_y_t(q_t_x)
        distortions = np.asarray([divergence(a, b) for a in p_y_x for b in q_y_t]).reshape(q_y_t.shape)
        return distortions

    rd, q_xt = blahut_arimoto(p_x=p_x,
                              beta=beta,
                              distortion=distortion,
                              max_iters=max_iters,
                              restarts=restarts
                              )

    sums = q_xt.sum(axis=1, keepdims=True)
    q_t_x = q_xt / sums
    q_t_x[np.isnan(q_t_x)] = 1 / np.array(q_t_x.shape)[sums.flatten() == 0]

    q_xyt = p_xy[:, :, np.newaxis] * q_t_x[:, np.newaxis, :]

    return rd, q_xyt


###############################################################################
# TODO: Deterministic Forms
