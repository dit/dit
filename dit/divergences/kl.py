import dit
import numpy as np

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
