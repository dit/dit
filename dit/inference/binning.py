"""
Various methods for binning real-valued data.
"""

from boltons.iterutils import pairwise

import numpy as np

__all__ = ['binned']


def binned(ts, bins=2, style='maxent'):
    """
    Discretize a real-valued list.

    Parameters
    ----------
    ts : ndarray
        The real-valued array to bin
    bins : int
        The number of bins to map the data into.
    style : str, {'maxent', 'uniform'}
        The method of discretizing the data. Defaults to 'maxent'.

    Returns
    -------
    symb : ndarray
        The discretized time-series.

    Raises
    ------
    ValueError
        Raised if `style` is not a recognized method.
    """
    if style == 'maxent':
        method = maxent_binning
    elif style == 'uniform':
        method = uniform_binning
    else: # pragma: no cover
        msg = "The style {} is not understood.".format(style)
        raise ValueError(msg)

    try:
        len(ts[0])
        one_d = False
    except:
        one_d = True

    ts = np.atleast_2d(ts)

    if one_d:
        ts = ts.T

    ts = np.vstack([method(ts[:, col], bins) for col in range(ts.shape[1])]).T

    if one_d:
        ts = ts.flatten()

    return ts

def uniform_binning(ts, bins):
    """
    Discretizes the time-series in to equal-width bins.

    Parameters
    ----------
    ts : ndarray
        The real-valued array to bin
    bins : int
        The number of bins to map the data into.

    Returns
    -------
    symb : ndarray
        The discretized time-series.
    """
    symb = np.asarray(bins*(ts - ts.min())/(ts.max() - ts.min() + 1e-12), dtype=int)
    return symb


def maxent_binning(ts, bins):
    """

    Parameters
    ----------
    ts : ndarray
        The real-valued array to bin
    bins : int
        The number of bins to map the data into.

    Returns
    -------
    symb : ndarray
        The discretized time-series.
    """
    symb = np.full_like(ts, np.nan)

    percentiles = np.percentile(ts, [100*i/bins for i in range(bins+1)])

    # Sometimes with large magnetude values things get weird. This helps:
    percentiles[0] = -np.inf
    percentiles[-1] = np.inf

    for i, (a, b) in enumerate(pairwise(percentiles)):
        symb[(a <= ts) & (ts < b)] = i

    symb = symb.astype(int)

    return symb
