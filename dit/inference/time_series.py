"""
Infer distributions from time series.
"""

import numpy as np

from .. import modify_outcomes
from ._symbols import Trials, is_trials
from .counts import distribution_from_data
from .markov_order import select_markov_order

__all__ = ("dist_from_timeseries",)


def _as_rows(observations):
    observations = np.atleast_2d(observations)
    if observations.shape[0] == 1:
        observations = observations.T
    return list(map(tuple, observations))


def dist_from_timeseries(observations, history_length=1, base="linear", max_history=None, prng=None):
    """
    Infer a distribution from time series observations. For each variable, infer a
    `history_length` past and a single observation present.

    Parameters
    ----------
    observations : list of tuples, ndarray, Trials
        A sequence of observations in time order, or independent
        :class:`~dit.inference.Trials` whose windows are pooled without any
        window spanning two trials.
    history_length : int or str
        The history length to utilize. If a string, the history length is the
        Markov order of the joint series estimated by
        :func:`~dit.inference.select_markov_order`: ``'auto'`` (the exact
        surrogate test) or one of its methods ``'exact'``, ``'chi2'``,
        ``'aic'``, ``'bic'``.
    base : float, str
        The base to use for the distribution. Defaults to 'linear'.
    max_history : int, None
        The largest history length considered when `history_length` is a
        string. Defaults to the largest ``L`` with ``M**(L + 1) <= N / 5``,
        for ``M`` joint symbols and ``N`` observations.
    prng : None, int, Generator, RandomState
        Source of randomness for ``'auto'`` / ``'exact'``.

    Returns
    -------
    ts : Distribution
        A distribution with the first half of the indices as the pasts
        of the various time series, and the second half their present values.
    """
    if is_trials(observations):
        observations = Trials(_as_rows(t) for t in observations)
        rows = [row for t in observations for row in t]
    else:
        observations = _as_rows(observations)
        rows = observations

    num_ts = len(rows[0])

    if isinstance(history_length, str):
        method = "exact" if history_length == "auto" else history_length
        if max_history is None:
            M = max(len(set(rows)), 2)
            max_history = max(int(np.log(len(rows) / 5) / np.log(M)) - 1, 0)
        history_length = select_markov_order(observations, max_history, method=method, prng=prng)

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
