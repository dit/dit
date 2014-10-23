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
