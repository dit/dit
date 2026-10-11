"""
The lautum (mutual backwards) information, as defined by Palomar & Verdu.
"""

import numpy as np

from dit.distconst import product_distribution
from dit.divergences import kullback_leibler_divergence
from dit.helpers import as_linear, normalize_rvs
from dit.utils import unitful

__all__ = ("lautum_information",)


@unitful
def lautum_information(dist, rvs=None, crvs=None):
    """
    Computes the lautum information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the lautum information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the lautum information. If None, then the
        lautum information is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    L : float
        The lautum information.

    Examples
    --------
    >>> outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
    >>> pmf = [3/16, 1/16, 1/16, 3/16, 1/16, 3/16, 3/16, 1/16]
    >>> d = dit.Distribution(outcomes, pmf)
    >>> dit.other.lautum_information(d)
    0.20751874963942196
    >>> dit.other.lautum_information(d, rvs=[[0], [1]])
    0.0

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution or if `rvs` or `crvs`
        contain non-existent random variables.
    """
    dist = as_linear(dist)
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    if not crvs:
        pd = product_distribution(dist, rvs=rvs)
        return kullback_leibler_divergence(pd, dist, rvs=rvs)

    # L(X;Y|Z) = D(p(z) prod_i p(x_i|z) || p(x, z)), Palomar & Verdu eq. (38).
    d = dist.coalesce(rvs + [crvs])
    d.make_dense()
    p = d.pmf.reshape([len(a) for a in d.alphabet])
    k = len(rvs)
    p_z = p.sum(axis=tuple(range(k)))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.ones_like(p)
        for i in range(k):
            others = tuple(j for j in range(k) if j != i)
            q = q * np.expand_dims(p.sum(axis=others), others)
        q = np.where(p_z > 0, q / p_z ** (k - 1), 0.0)
        if np.any((q > 0) & (p == 0)):
            return np.inf
        terms = np.where(q > 0, q * np.log2(q / p), 0.0)
    return float(terms.sum())
