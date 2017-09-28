"""
"""
from __future__ import division

import numpy as np
from scipy.special import digamma


try: # cython
    from . import counts_from_data
    def get_counts(data, length):
        """
        Count the occurrences of all words of `length` in `data`.

        Parameters
        ----------
        data : iterable
            The sequence of samples
        length : int
            The length to group samples into.

        Returns
        -------
        counts : np.array
            Array with the count values.

        Notes
        -----
        This function utilizes the cython-implemented `counts_from_data`.
        """
        counts = counts_from_data(data, 0, length)[1][0]
        return counts

except ImportError: # no cython
    from collections import Counter
    from boltons.iterutils import windowed_iter
    def get_counts(data, length):
        """
        Count the occurrences of all words of `length` in `data`.

        Parameters
        ----------
        data : iterable
            The sequence of samples
        length : int
            The length to group samples into.

        Returns
        -------
        counts : np.array
            Array with the count values.

        Notes
        -----
        This function uses `collections.Counter` and `boltons.iterutils.windowed_iter`.
        """
        counts = Counter(windowed_iter(data, length))
        counts = np.array(list(counts.values()))
        return counts


def entropy_0(data, length=1):
    """
    Estimate the entropy of length `length` subsequences in `data`.

    Parameters
    ----------
    data : iterable
        An iterable of samples.
    length : int
        The length to group samples into.

    Returns
    -------
    h0 : float
        An estimate of the entropy.

    Notes
    -----
    This returns the naive estimate of the entropy.
    """
    counts = get_counts(data, length)
    probs = counts/counts.sum()
    h0 = -np.nansum(probs * np.log2(probs))
    return h0


def entropy_1(data, length=1):
    """
    Estimate the entropy of length `length` subsequences in `data`.

    Parameters
    ----------
    data : iterable
        An iterable of samples.
    length : int
        The length to group samples into.

    Returns
    -------
    h0 : float
        An estimate of the entropy.

    Notes
    -----
    This returns a less naive estimate of the entropy.
    """
    counts = get_counts(data, length)
    total = counts.sum()
    digamma_N = digamma(total)

    h1 = np.log2(np.e)*(counts/total*(digamma_N - digamma(counts))).sum()

    return h1


def entropy_2(data, length=1):
    """
    Estimate the entropy of length `length` subsequences in `data`.

    Parameters
    ----------
    data : iterable
        An iterable of samples.
    length : int
        The length to group samples into.

    Returns
    -------
    h0 : float
        An estimate of the entropy.

    Notes
    -----
    This returns a bias-corrected estimate of the entropy.
    """
    counts = get_counts(data, length)
    total = counts.sum()
    digamma_N = digamma(total)
    log2 = np.log(2)
    jss = [np.arange(1, count) for count in counts]

    alt_terms = np.array([(((-1)**js)/js).sum() for js in jss])

    h2 = np.log2(np.e)*(counts/total*(digamma_N - digamma(counts) + log2 + alt_terms)).sum()

    return h2
