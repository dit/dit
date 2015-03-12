"""
These are special interest implementations that should be used only in
very particular situations.

cross_entropy_pmf
relative_entropy_pmf
DKL_pmf
    These functions should be used only if the pmfs that are passed in have
    the same exact length and correspond to the same outcome probabilities.
    They also assume linear distributed probabilities. The expected use case
    is when one is working with a family of distributions (or pmfs) all of
    which have the same exact sample space. For example, normally distributed
    pmfs can be generated as: dit.math.norm([.5, .5], size=5). You can
    pass those distributions to sklearn.metrics.pairwise_distance with
    metric=DKL_pmf.

cross_entropy
relative_entropy
DKL
    These functions should be used only if the sample spaces of the passed in
    distributions are identical (so both the same size and the same order).
    The two distributions can have pmfs in different bases.

"""

import dit
import numpy as np

def cross_entropy_pmf(p, q=None):
    """
    Calculates the cross entropy from probability mass functions `p` and `q`.

    If `q` is None, then it is set to be `p`.
    Then the entropy of `p` is calculated.

    Assumption: Linearly distributed probabilities.

    """
    if q is None:
        q = p

    p = np.asarray(p)
    q = np.asarray(q)

    return -np.nansum(p * np.log2(q))

entropy_pmf = cross_entropy_pmf

def relative_entropy_pmf(p, q):
    """
    Calculates the relative entropy (or Kullback-Leibler divergence).

    Assumption: Linearly distributed probabilities.

    .. math::

        D_{KL}(p || q)

    """
    return cross_entropy_pmf(p, q) - cross_entropy_pmf(p, p)

DKL_pmf = relative_entropy_pmf

def cross_entropy(d1, d2, pmf_only=True):
    """
    Returns H(d1, d2)

    """
    if pmf_only:
        mode = 'asis'
    else:
        mode = 'dense'

    pmf1 = dit.copypmf(d1, base='linear', mode=mode)
    pmf2 = dit.copypmf(d2, base='linear', mode=mode)
    return -np.nansum(pmf1 * np.log2(pmf2))

def relative_entropy(d1, d2):
    ce = cross_entropy(d1, d2, pmf_only=False)
    return ce - dit.shannon.entropy(d1)

DKL = relative_entropy
