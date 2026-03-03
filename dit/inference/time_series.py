"""
Infer distributions from time series.
"""

import numpy as np

from .. import modify_outcomes
from .counts import distribution_from_data

__all__ = ("dist_from_timeseries",)


def dist_from_timeseries(observations, history_length=1, base="linear"):
    """
    Infer a distribution from time series observations. For each variable, infer a
    `history_length` past and a single observation present.

    Parameters
    ----------
    observations : list of tuples, ndarray
        A sequence of observations in time order.
    history_length : int
        The history length to utilize.
    base : float, str
        The base to use for the distribution. Defaults to 'linear'.

    Returns
    -------
    ts : Distribution
        A distribution with the first half of the indices as the pasts
        of the various time series, and the second half their present values.
    """
    observations = np.atleast_2d(observations)
    if observations.shape[0] == 1:
        observations = observations.T
    observations = list(map(tuple, observations))

    num_ts = len(observations[0])

    d = distribution_from_data(observations, L=history_length + 1, base=base)

    if history_length > 0 and num_ts > 1:
        # Reorder from time-interleaved (v1_t0, v2_t0, v1_t1, v2_t1, ...)
        # to variable-grouped (v1_t0, v1_t1, ..., v2_t0, v2_t1, ..., presents)
        def f(o):
            steps = [o[i * num_ts : (i + 1) * num_ts] for i in range(history_length + 1)]
            pasts = tuple(steps[t][v] for v in range(num_ts) for t in range(history_length))
            presents = steps[-1]
            return pasts + presents

        d = modify_outcomes(d, f)

    return d
