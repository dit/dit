"""
Various methods for binning real-valued data.
"""

import numpy as np
from boltons.iterutils import pairwise

__all__ = (
    "binned",
    "ordinal_patterns",
    "relative_rank",
)


def binned(ts, bins=2, style="maxent"):
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
    if style == "maxent":
        method = maxent_binning
    elif style == "uniform":
        method = uniform_binning
    else:  # pragma: no cover
        msg = f"The style {style} is not understood."
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
    symb = np.asarray(bins * (ts - ts.min()) / (ts.max() - ts.min() + 1e-12), dtype=int)
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

    percentiles = np.percentile(ts, [100 * i / bins for i in range(bins + 1)])

    # Sometimes with large magnetude values things get weird. This helps:
    percentiles[0] = -np.inf
    percentiles[-1] = np.inf

    for i, (a, b) in enumerate(pairwise(percentiles)):
        symb[(a <= ts) & (ts < b)] = i

    symb = symb.astype(int)

    return symb


def _windows(ts, length, delay):
    ts = np.asarray(ts, dtype=float)
    n = len(ts) - (length - 1) * delay
    if n <= 0:
        return np.zeros((0, length))
    return np.stack([ts[k * delay : k * delay + n] for k in range(length)], axis=1)


def _break_ties(windows, ties, prng):
    if ties == "noise":
        from ._symbols import as_generator

        scale = np.abs(windows).max() if windows.size else 1.0
        return windows + as_generator(prng).uniform(0, 1e-10 * max(scale, 1.0), size=windows.shape)
    if ties not in ("first", "distinct"):
        raise ValueError(f"Unknown ties rule {ties!r}; use 'first', 'noise', or 'distinct'.")
    return windows


def _lehmer(perms):
    """
    Index of each permutation (row) in the lexicographic order of all permutations.
    """
    n, m = perms.shape
    codes = np.zeros(n, dtype=np.int64)
    factorial = 1
    for i in range(m - 1, -1, -1):
        smaller = np.sum(perms[:, i + 1 :] < perms[:, i : i + 1], axis=1)
        codes += smaller * factorial
        factorial *= m - i
    return codes


def ordinal_patterns(ts, order=3, delay=1, ties="first", prng=None):
    """
    Map each delay window of a real-valued series to its ordinal pattern.

    The window ending at time ``t`` is
    :math:`(x_{t-(m-1)\\tau}, \\ldots, x_{t-\\tau}, x_t)` with :math:`m` = `order` and
    :math:`\\tau` = `delay`; its pattern is the permutation that sorts it
    :cite:`Bandt2002`. Patterns are invariant to monotone transforms of the series.

    Parameters
    ----------
    ts : array_like or Trials
        A real-valued series, or independent :class:`~dit.inference.Trials`.
    order : int
        The window length :math:`m \\geq 2`.
    delay : int
        The spacing :math:`\\tau \\geq 1` between window elements.
    ties : {'first', 'noise', 'distinct'}
        How equal values are ranked. ``'first'`` ranks the earlier occurrence
        lower (the usual convention); ``'noise'`` adds negligible random noise;
        ``'distinct'`` keeps ties as their own patterns :cite:`Bian2012`.
    prng : None, int, Generator, RandomState
        Source of randomness for ``ties='noise'``.

    Returns
    -------
    patterns : np.ndarray or Trials
        Integer pattern codes, one per window (``len(ts) - (order - 1) * delay``);
        entry ``i`` is the window ending at ``t = i + (order - 1) * delay``. With
        ``'first'`` or ``'noise'`` codes are the lexicographic index of the
        permutation, in ``0 .. order! - 1``. With ``'distinct'`` they are base-`order`
        codes of the dense rank vector.
    """
    from ._symbols import Trials, is_trials

    if is_trials(ts):
        return Trials(ordinal_patterns(t, order, delay, ties, prng) for t in ts)
    if order < 2 or delay < 1:
        raise ValueError("`order` must be at least 2 and `delay` at least 1.")
    windows = _break_ties(_windows(ts, order, delay), ties, prng)
    if ties == "distinct":
        ranks = np.stack([np.unique(w, return_inverse=True)[1].ravel() for w in windows]) if len(windows) else windows
        return (ranks.astype(np.int64) * order ** np.arange(order)[::-1]).sum(axis=1)
    perms = np.argsort(np.argsort(windows, axis=1, kind="stable"), axis=1, kind="stable")
    return _lehmer(perms)


def relative_rank(ts, order=3, delay=1, ties="first", prng=None):
    """
    The rank of each value among the `order` values preceding it.

    For time ``t`` this is the number of values among
    :math:`x_{t-\\tau}, \\ldots, x_{t-m\\tau}` that are smaller than :math:`x_t` (or,
    with ``ties='first'``, not larger), in ``0 .. order``. Paired with
    :func:`ordinal_patterns` of the preceding window, it encodes the present of a
    series without sharing values with its past, which removes the leakage that
    biases symbolic transfer entropy :cite:`Kugiumtzis2012`.

    Parameters
    ----------
    ts : array_like or Trials
        A real-valued series, or independent trials.
    order : int
        The number :math:`m` of preceding values.
    delay : int
        The spacing :math:`\\tau` between them.
    ties : {'first', 'noise', 'distinct'}
        ``'first'`` counts earlier equal values as smaller; ``'distinct'`` counts
        only strictly smaller values; ``'noise'`` adds negligible noise first.
    prng : None, int, Generator, RandomState
        Source of randomness for ``ties='noise'``.

    Returns
    -------
    ranks : np.ndarray or Trials
        Entry ``i`` is the rank of :math:`x_t` with ``t = i + order * delay``.
    """
    from ._symbols import Trials, is_trials

    if is_trials(ts):
        return Trials(relative_rank(t, order, delay, ties, prng) for t in ts)
    if order < 1 or delay < 1:
        raise ValueError("`order` and `delay` must be positive.")
    windows = _break_ties(_windows(ts, order + 1, delay), ties, prng)
    present = windows[:, -1:]
    past = windows[:, :-1]
    if ties == "distinct":
        return np.sum(past < present, axis=1).astype(np.int64)
    return np.sum(past <= present, axis=1).astype(np.int64)
