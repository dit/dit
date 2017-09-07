"""
"""
from __future__ import division

import numpy as np
from scipy.special import digamma


try: # cython
    from . import counts_from_data
    def get_counts(data, length):
        """

        Parameters
        ----------
        data
        length

        Returns
        -------

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

        Parameters
        ----------
        data
        length

        Returns
        -------

        Notes
        -----
        This function uses `collections.Counter` and `boltons.iterutils.windowed_iter`.
        """
        counts = Counter(windowed_iter(data, length))
        counts = np.array(counts.values())
        return counts


def entropy_0(data, length=1):
    """
    """
    counts = get_counts(data, 0, length)
    probs = counts[0]/counts[0].sum()
    h0 = np.nansum(probs * np.log2(probs))
    return h0


def entropy_1(data, length=1):
    """
    """
    counts = get_counts(data, 0, length)
    total = counts.sum()
    digamma_N = digamma(total)

    h1 = np.log2(np.e)*(counts/total*(digamma_N - digamma(counts))).sum()

    return h1


def entropy_2(data, length=1):
    """
    """
    counts = get_counts(data, 0, length)
    total = counts.sum()
    digamma_N = digamma(total)
    log2 = np.log(2)
    jss = [np.arange(1, count) for count in counts[0]]

    alt_terms = np.array([(((-1)**js)/js).sum() for js in jss])

    h2 = np.log2(np.e)*(counts/total*(digamma_N - digamma(counts) + log2 + alt_terms)).sum()

    return h2


def conditional_mutual_information_ksg():
    """
    """
    pass
