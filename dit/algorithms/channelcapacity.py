"""
Simple implementation for channel capacity.

"""

import numpy as np
import dit

from ..exceptions import ditException
from ..cdisthelpers import cdist_array


__all__ = [
    'channel_capacity',
]

def channel_capacity(cdists, marginal=None, rtol=None, atol=None):
    """
    Calculates the channel capacity from conditional distributions P(Y|X).

    Parameters
    ----------
    cdists : list, ndarray
        A list of conditional distributions. For each ``x=outcomes[i]``, the
        corresponding element ``cdists[i]`` should represent P(Y | X=x); or
        the conditional distribution as an array.
    marginal : distribution | None
        The marginal distribution P(X) that goes with P(Y|X). This is optional,
        but if provided, then it is used to construct a Distribution object
        for the distribution P*(X) which achieves the channel capacity.
        If ``None`` then P*(X) is returned as a NumPy array.
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
    >>> d = dit.random_distribution(3, 2)
    >>> mdist_true, cdists = d.condition_on([1])
    >>> cc, mdist_opt = channel_capacity(cdists, mdist_true)

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

    try:
        cdists.shape
        carr = cdists
    except AttributeError:
        # Build the array for P(Y|X)
        # We need Y dense so that we search the appropriate space.
        carr = cdist_array(cdists, base='linear', mode='dense')

    if marginal and len(marginal) != carr.shape[0]:
        msg = 'len(mdist) != len(cdists)'
        raise ditException(msg)

    cc_iter = next_cc(carr)
    cc, pmf = next(cc_iter)
    old_cc = 0
    while not np.isclose(cc, old_cc, rtol=rtol, atol=atol):
        old_cc, (cc, pmf) = cc, next(cc_iter)

    if marginal is not None:
        marginal_opt = marginal.copy()
        marginal_opt.pmf = pmf
    else:
        marginal_opt = pmf

    return cc, marginal_opt


def channel_capacity_joint(dist, input, output, rv_mode=None, marginal=False):
    """
    Compute the channel capacity from ``rvs1`` to ``rvs2``.

    Parameters
    ----------
    dist : Distribution
        The distribution to find a channel capacity within.
    input : iterable
        The random variables that are the input of the channel.
    output : iterable
        The random variables that are the output of the channel.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------

    """
    _, cdists = dist.condition_on(crvs=input, rvs=output, rv_mode=rv_mode)
    cc, marg_opt = channel_capacity(cdists)
    if marginal:
        return cc, marg_opt
    else:
        return cc
