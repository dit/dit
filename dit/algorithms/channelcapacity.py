"""
Simple implementation for channel capacity.

"""

import numpy as np
import dit

from ..cdisthelpers import (
    cdist_array,
    mask_is_complementary,
    joint_from_factors,
)

__all__ = [
    'channel_capacity',
]

def channel_capacity(cdists, marginal=None, rtol=None, atol=None):
    """
    Calculates the channel capacity from conditional distributions P(Y|X).

    Parameters
    ----------
    cdists : list
        A list of conditional distributions. For each ``x=outcomes[i]``, the
        corresponding element ``cdists[i]`` should represent P(Y | X=x).
    marginal : dist | None
        The marginal distribution P(X) that goes with P(Y|X). This is optional,
        but if provided, then it is used to construct a Distribution object
        for the distribution P*(X) which achieves the channel capacity.
        If ``None`` then P(X) is returned as a NumPy array.
    rtol : None
        Relative tolerance used to determine convergence criterion. This is
        passed to ``np.isclose``.
    atol : None
        Absolute tolerance used to determine convergence criterion. This is
        passed to ``np.isclose``.

    Returns
    -------
    cc : float
        The channel capacity.
    marginal_opt : Distribution
        The optimal marginal distribution for P(X) which achieves the
        channel capacity by maximizing the mutual information I[X:Y].

    Examples
    --------
    >>> d = dit.random_distribution(2, 2)
    >>> marginal_true, cdists = d.condition_on([0])
    >>> cc, marginal_opt = channel_capacity(marginal_true.outcomes, cdists)

    """
    if rtol is None:
        rtol = dit.ditParams['rtol']
    if atol is None:
        atol = dit.ditParams['atol']

    def next_r(p, q):
        r = (q ** p.T).prod(axis=0)
        r /= r.sum()
        return r

    def next_q(p, r):
        q = r * p.T
        q /= q.sum(1)[:, np.newaxis]
        return q

    def calc_cc(p, q, r):
        tmp = r * p.T * np.log2(q / r)
        return np.nansum(tmp)

    def next_cc(p):
        r = np.ones(p.shape[0], dtype=float)
        r /= r.sum()
        while True:
            q = next_q(p, r)
            r = next_r(p, q)
            cc = calc_cc(p, q, r)
            yield cc, r

    # Build the array for P(Y|X)
    carr = cdist_array(cdists)

    if marginal and len(marginal) != carr.shape[0]:
        msg = 'len(mdist) != len(cdists)'
        raise ditException(msg)

    cc_iter = next_cc(carr)
    cc, pmf = cc_iter.next()
    old_cc = 0
    while not np.isclose(cc, old_cc, rtol=rtol, atol=atol):
        old_cc, (cc, pmf) = cc, cc_iter.next()

    if marginal is not None:
        marginal_opt = marginal.copy()
        marginal_opt.pmf = pmf
    else:
        marginal_opt = pmf

    return cc, marginal_opt
