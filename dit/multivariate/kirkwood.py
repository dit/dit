"""
Mutual informations defined as the divergence from a joint distribution to a
normalized log-linear product of its marginals: the generalized Kirkwood
superposition approximation and the order-k ouroboros approximation.
"""

from itertools import combinations
from math import comb

import numpy as np

from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..utils import unitful

__all__ = (
    "kirkwood_mutual_information",
    "ouroboros_mutual_information",
)


def _joint_array(dist, rvs, crvs):
    """
    Coalesce `dist` into a dense array with one axis per element of `rvs`
    followed by a single axis for `crvs` (of size one when unconditioned).
    """
    groups = list(rvs) + ([crvs] if crvs else [])
    d = dist.coalesce(groups)
    d.make_dense()
    pmf = d.pmf.reshape([len(a) for a in d.alphabet])
    if not crvs:
        pmf = pmf[..., np.newaxis]
    return pmf


def _log_linear_divergence(pmf, exponents):
    """
    Compute sum_z p(z) D( p(. | z) || q_z ), where q_z is proportional to
    prod_S p(x_S | z)^{c_S}.

    Parameters
    ----------
    pmf : np.ndarray
        Joint pmf with one axis per variable and a final conditioning axis.
    exponents : dict
        Maps tuples of variable axes S to the exponent c_S.

    Returns
    -------
    dkl : float
        The averaged divergence, in bits.
    """
    n = pmf.ndim - 1
    axes = tuple(range(n))
    p_z = pmf.sum(axis=axes, keepdims=True)
    safe_p_z = np.where(p_z > 0, p_z, 1.0)

    log_q = np.zeros(pmf.shape)
    zero = np.zeros(pmf.shape, dtype=bool)
    for subset, exponent in exponents.items():
        others = tuple(i for i in axes if i not in subset)
        marginal = pmf.sum(axis=others, keepdims=True)
        zero |= np.broadcast_to(marginal == 0, pmf.shape)
        log_q += exponent * np.log2(np.where(marginal > 0, marginal, 1.0) / safe_p_z)

    log_q = np.where(zero, -np.inf, log_q)
    shift = np.max(log_q, axis=axes, keepdims=True)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    support = pmf > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_Z = shift + np.log2(np.exp2(log_q - shift).sum(axis=axes, keepdims=True))
        log_p_cond = np.log2(np.where(support, pmf, 1.0) / safe_p_z)
        log_ratio = np.broadcast_to(log_p_cond - (log_q - log_Z), pmf.shape)
    dkl = (pmf[support] * log_ratio[support]).sum()

    return max(float(dkl), 0.0)


@unitful
def kirkwood_mutual_information(dist, rvs=None, crvs=None):
    r"""
    Compute the Kirkwood mutual information: the Kullback-Leibler divergence
    from the joint distribution to its normalized generalized Kirkwood
    superposition approximation.

    For variables :math:`X_0, \ldots, X_{n-1}` the generalized Kirkwood
    approximation is the Möbius product over proper subsets
    :cite:`watanabe1960information`,

    .. math::
        \tilde{p}(x) = \prod_{\emptyset \neq S \subsetneq [n]} p(x_S)^{(-1)^{n-1-|S|}},

    which need not sum to one. With :math:`\hat{p} = \tilde{p} / Z`,
    :math:`Z = \sum_x \tilde{p}(x)`, the measure is
    :math:`K = D_{KL}(p \| \hat{p})`. For three variables this is the
    divergence from the normalized Kirkwood superposition approximation used
    by :cite:`wan2010boost`. It satisfies

    .. math::
        K = (-1)^n I[X_0 : \cdots : X_{n-1}] + \log_2 Z,

    where :math:`I` is the coinformation :cite:`kubkowski2020asymptotic`.
    For two variables it is the mutual information; for a single variable it
    is zero.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the Kirkwood mutual information is
        calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the Kirkwood mutual information. If None,
        then it is calculated over all random variables.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    K : float
        The Kirkwood mutual information.

    Examples
    --------
    >>> d = dit.example_dists.giant_bit(3, 2)
    >>> dit.multivariate.kirkwood_mutual_information(d)
    0.0
    >>> d = dit.example_dists.n_mod_m(3, 2)
    >>> dit.multivariate.kirkwood_mutual_information(d)
    1.0
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    n = len(rvs)
    if n < 2:
        return 0.0

    exponents = {S: (-1) ** (n - 1 - k) for k in range(1, n) for S in combinations(range(n), k)}

    return _log_linear_divergence(_joint_array(dist, rvs, crvs), exponents)


@unitful
def ouroboros_mutual_information(dist, order=None, rvs=None, crvs=None):
    r"""
    Compute the order-k ouroboros mutual information: the Kullback-Leibler
    divergence from the joint distribution to its normalized order-k
    ouroboros approximation.

    The order-k ouroboros approximation of :math:`X_0, \ldots, X_{n-1}` is the
    symmetric geometric mean, over all wirings in which each variable is the
    output of a channel fed by k other variables, of the resulting products of
    conditional distributions:

    .. math::
        \tilde{p}_k(x) = \frac{\prod_{|S| = k+1} p(x_S)^{n / \binom{n}{k+1}}}
                              {\prod_{|S| = k} p(x_S)^{n / \binom{n}{k}}},
        \qquad 1 \leq k \leq n - 2.

    With :math:`\hat{p}_k = \tilde{p}_k / Z` the measure is
    :math:`O_k = D_{KL}(p \| \hat{p}_k)`. For three variables and k = 1 it
    coincides with the Kirkwood mutual information. For k = 1 the
    approximation is the :math:`(n-1)`-th root of the pairwise conditional
    composite likelihood :cite:`varin2011overview`. Since :math:`\hat{p}_k`
    is log-linear in the :math:`(k+1)`-marginals, :math:`O_k` upper bounds the
    divergence to the maximum entropy distribution matching those marginals.

    No canonical literature source is known for this construction.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the ouroboros mutual information is
        calculated.
    order : int, None
        The number of inputs k to each channel. Must satisfy
        1 <= k <= n - 2. If None, defaults to n - 2.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used. If None, then all random variables are used.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    O : float
        The ouroboros mutual information.

    Raises
    ------
    ditException
        Raised if fewer than three variables are given or if `order` is out
        of range.

    Examples
    --------
    >>> d = dit.example_dists.giant_bit(4, 2)
    >>> dit.multivariate.ouroboros_mutual_information(d, order=1)
    0.0
    >>> d = dit.example_dists.n_mod_m(4, 2)
    >>> dit.multivariate.ouroboros_mutual_information(d)
    1.0
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    n = len(rvs)
    if n < 3:
        msg = f"The ouroboros mutual information requires at least 3 variables, {n} given."
        raise ditException(msg)

    k = n - 2 if order is None else order
    if not 1 <= k <= n - 2:
        msg = f"order must satisfy 1 <= order <= {n - 2}, {k} given."
        raise ditException(msg)

    exponents = {S: n / comb(n, k + 1) for S in combinations(range(n), k + 1)}
    exponents.update({S: -n / comb(n, k) for S in combinations(range(n), k)})

    return _log_linear_divergence(_joint_array(dist, rvs, crvs), exponents)
