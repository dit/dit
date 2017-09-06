"""
"""
from __future__ import division

import numpy as np
from scipy.special import digamma

from . import counts_from_data

def entropy_naive(data, length=1):
    """
    """
    _, counts, _, _ = counts_from_data(data, 0, length)
    probs = counts[0]/counts[0].sum()
    h1 = np.nansum(probs * np.log2(probs))
    return h1

def entropy_bias_corrected(data, length=1):
    """
    """
    _, counts, _, _ = counts_from_data(data, 0, length)
    total = counts.sum()
    digamma_N = digamma(total)
    log2 = np.log(2)
    jss = [np.arange(1, count) for count in counts[0]]

    alt_terms = np.array([(((-1)**js)/js).sum() for js in jss])

    h2 = np.log2(np.e)*(counts/total*(digamma_N - digamma(counts) + log2 + alt_terms)).sum()

    return h2

def mutual_information_ksg():
    """
    """
    pass
