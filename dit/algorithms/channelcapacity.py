"""
Simple implementation for channel capacity.
"""

import numpy as np

import dit

from ..cdisthelpers import cdist_array
from ..exceptions import ditException
from ..utils import unitful

__all__ = ("channel_capacity",)


def channel_capacity(cdists, marginal=None, rtol=None, atol=None):
    """
    Calculates the channel capacity from conditional distributions P(Y|X).

    Parameters
    ----------
    cdists : list, ndarray, Distribution
        A list of conditional distributions, a 2D array where rows are
        inputs and columns are outputs, or a conditional Distribution
        such as p(Y|X).
    marginal : distribution | None
        The marginal distribution P(X). If provided, P*(X) is returned as
        a distribution object. If None, P*(X) is returned as a numpy array.
    rtol : float or None
        Relative convergence tolerance. Defaults to ditParams['rtol'].
    atol : float or None
        Absolute convergence tolerance. Defaults to ditParams['atol'].

    Returns
    -------
    cc : float
        The channel capacity.
    marginal_opt : distribution or ndarray
        The optimal marginal P*(X).
    """
    from ..distribution import Distribution

    if rtol is None:
        rtol = dit.ditParams["rtol"]
    if atol is None:
        atol = dit.ditParams["atol"]

    def next_r(p, q):
        r = (q**p.T).prod(axis=0)
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

    is_xr = isinstance(cdists, Distribution)
    is_xr_conditional = is_xr and cdists.is_conditional()

    if is_xr and not is_xr_conditional:
        raise ditException("Distribution passed to channel_capacity must be conditional")

    if is_xr_conditional:
        lin = cdists._linear_data()
        given_dims = [d for d in cdists.dims if d in cdists.given_vars]
        free_dims = [d for d in cdists.dims if d in cdists.free_vars]
        reordered = lin.transpose(*given_dims, *free_dims)
        n_given = int(np.prod([len(cdists.data.coords[d]) for d in given_dims]))
        n_free = int(np.prod([len(cdists.data.coords[d]) for d in free_dims]))
        carr = reordered.values.reshape(n_given, n_free)
    else:
        try:
            cdists.shape  # noqa: B018
            carr = cdists
        except AttributeError:
            carr = cdist_array(cdists, base="linear", mode="dense")

    if marginal is not None:
        n_inputs = len(marginal)
        if n_inputs != carr.shape[0]:
            msg = "len(mdist) != len(cdists)"
            raise ditException(msg)

    cc_iter = next_cc(carr)
    cc, pmf = next(cc_iter)
    old_cc = 0
    while not np.isclose(cc, old_cc, rtol=rtol, atol=atol):
        old_cc, (cc, pmf) = cc, next(cc_iter)

    if marginal is not None:
        if is_xr_conditional and isinstance(marginal, Distribution):
            result = marginal.copy()
            result.data = result.data.copy(deep=True)
            result.data.values[:] = pmf.reshape(result.data.shape)
            marginal_opt = result
        else:
            marginal_opt = marginal.copy()
            marginal_opt.pmf = pmf
    else:
        marginal_opt = pmf

    return cc, marginal_opt


@unitful
def channel_capacity_joint(dist, input_, output, marginal=False):
    """
    Compute the channel capacity from ``input_`` to ``output``.

    Parameters
    ----------
    dist : Distribution
        The joint distribution.
    input_ : iterable
        The random variables that are the input of the channel.
    output : iterable
        The random variables that are the output of the channel.
    marginal : bool
        Whether to return the marginal distribution. Defaults to False.
    """
    input_names = list(dist._resolve_rv_names(list(input_)))
    output_names = list(dist._resolve_rv_names(list(output)))

    keep_vars = input_names + output_names
    sub = dist.marginal(*keep_vars)
    marg, cdist_list = sub.condition_on(input_names)
    cc, marg_opt = channel_capacity(cdist_list, marg)
    if marginal:
        return cc, marg_opt
    else:
        return cc
