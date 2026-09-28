"""
Some estimators based on estimating local densities using k-nearest neighbors.
"""

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[attr-defined]
from scipy.special import digamma

from dit.utils import flatten

from ._symbols import as_generator

__all__ = (
    "conditional_mutual_information_test_knn",
    "differential_entropy_knn",
    "total_correlation_ksg",
)


def _fuzz(data, noise, prng=None):
    """
    Add noise to the data.

    Parameters
    ----------
    data : np.ndarray
        Data.
    noise : float
        The standard deviation of the normally-distributed noise to add to data.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    data : np.ndarray
        The fuzzed data.
    """
    data = data.astype(np.float64)
    data += as_generator(prng).normal(0.0, noise, size=data.shape)
    return data


def differential_entropy_knn(data, rvs=None, k=4, noise=1e-10, prng=None):
    """
    Compute the *differential* entropy of `data` using a k-nearest neighbors density estimator.

    Parameters
    ----------
    data : np.ndarray
        The data.
    rvs : list
        The columns of `data` to use as the random variable. If None, use all.
    k : int
        The number of nearest neighbors to use.
    noise : float
        The standard deviation of the normally-distributed noise to add to data.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    h : float
        The estimated entropy.

    Notes
    -----
    The entropy is returned in units of bits.
    """
    if rvs is None:
        rvs = list(range(data.shape[1]))

    data = _fuzz(data, noise, prng)

    d = len(rvs)

    tree = cKDTree(data[:, rvs])

    epsilons = tree.query(data[:, rvs], k + 1, p=np.inf)[0][:, -1]

    h = digamma(len(data)) - digamma(k) + d * (np.log(2) + np.log(epsilons).mean())

    return h / np.log(2)


def _total_correlation_ksg_scipy(data, rvs, crvs=None, k=4, noise=1e-10, prng=None):
    """
    Compute the total correlation from observations. The total correlation is computed between the columns
    specified in `rvs`, given the columns specified in `crvs`. This utilizes the KSG kNN density estimator,
    and works on discrete, continuous, and mixed data.

    Parameters
    ----------
    data : np.array
        A set of observations of a distribution.
    rvs : iterable of iterables
        The columns for which the total correlation is to be computed.
    crvs : iterable
        The columns upon which the total correlation should be conditioned.
    k : int
        The number of nearest neighbors to use in estimating the local kernel density.
    noise : float
        The standard deviation of the normally-distributed noise to add to the data.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    tc : float
        The total correlation of `rvs` given `crvs`.

    Notes
    -----
    The total correlation is computed in bits, not nats as most KSG estimators do.
    """
    # KSG suggest adding noise (to break symmetries?)
    data = _fuzz(data, noise, prng)

    if crvs is None:
        crvs = []

    digamma_N = digamma(len(data))
    log_2 = np.log(2)

    all_rvs = list(flatten(rvs)) + crvs
    rvs = [rv + crvs for rv in rvs]

    d_rvs = [len(data[0, rv]) for rv in rvs]

    tree = cKDTree(data[:, all_rvs])
    tree_rvs = [cKDTree(data[:, rv]) for rv in rvs]

    epsilons = tree.query(data[:, all_rvs], k + 1, p=np.inf)[0][:, -1]  # k+1 because of self

    n_rvs = [
        np.array(
            [
                len(t.query_ball_point(point, epsilon, p=np.inf))
                for point, epsilon in zip(data[:, rv], epsilons, strict=True)
            ]
        )
        for rv, t in zip(rvs, tree_rvs, strict=True)
    ]

    log_epsilons = np.log(epsilons)

    h_rvs = [-digamma(n_rv).mean() for n_rv, d in zip(n_rvs, d_rvs, strict=True)]

    h_all = -digamma(k)

    if crvs:
        tree_crvs = cKDTree(data[:, crvs])
        n_crvs = np.array(
            [
                len(tree_crvs.query_ball_point(point, epsilon, p=np.inf))
                for point, epsilon in zip(data[:, crvs], epsilons, strict=True)
            ]
        )
        h_crvs = -digamma(n_crvs).mean()
    else:
        h_rvs = [h_rv + digamma_N + d * (log_2 - log_epsilons).mean() for h_rv, d in zip(h_rvs, d_rvs, strict=True)]
        h_all += digamma_N + sum(d_rvs) * (log_2 - log_epsilons).mean()
        h_crvs = 0

    tc = sum(h_rv - h_crvs for h_rv in h_rvs) - (h_all - h_crvs)

    return tc / log_2


def _total_correlation_ksg_sklearn(data, rvs, crvs=None, k=4, noise=1e-10, prng=None):
    """
    Compute the total correlation from observations. The total correlation is computed between the columns
    specified in `rvs`, given the columns specified in `crvs`. This utilizes the KSG kNN density estimator,
    and works on discrete, continuous, and mixed data.

    Parameters
    ----------
    data : np.array
        Real valued time series data.
    rvs : iterable of iterables
        The columns for which the total correlation is to be computed.
    crvs : iterable
        The columns upon which the total correlation should be conditioned.
    k : int
        The number of nearest neighbors to use in estimating the local kernel density.
    noise : float
        The standard deviation of the normally-distributed noise to add to the data.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    tc : float
        The total correlation of `rvs` given `crvs`.

    Notes
    -----
    The total correlation is computed in bits, not nats as most KSG estimators do.

    This implementation uses scikit-learn.
    """
    # KSG suggest adding noise (to break symmetries?)
    data = _fuzz(data, noise, prng)

    if crvs is None:
        crvs = []

    digamma_N = digamma(len(data))
    log_2 = np.log(2)

    all_rvs = list(flatten(rvs)) + crvs
    rvs = [rv + crvs for rv in rvs]

    d_rvs = [len(data[0, rv]) for rv in rvs]

    tree = KDTree(data[:, all_rvs], metric="chebyshev")
    tree_rvs = [KDTree(data[:, rv], metric="chebyshev") for rv in rvs]

    epsilons = tree.query(data[:, all_rvs], k + 1)[0][:, -1]  # k+1 because of self

    n_rvs = [t.query_radius(data[:, rv], epsilons, count_only=True) for rv, t in zip(rvs, tree_rvs, strict=True)]

    log_epsilons = np.log(epsilons)

    h_rvs = [-digamma(n_rv).mean() for n_rv, d in zip(n_rvs, d_rvs, strict=True)]

    h_all = -digamma(k)

    if crvs:
        tree_crvs = KDTree(data[:, crvs], metric="chebyshev")
        n_crvs = tree_crvs.query_radius(data[:, crvs], epsilons, count_only=True)
        h_crvs = -digamma(n_crvs).mean()
    else:
        h_rvs = [h_rv + digamma_N + d * (log_2 - log_epsilons).mean() for h_rv, d in zip(h_rvs, d_rvs, strict=True)]
        h_all += digamma_N + sum(d_rvs) * (log_2 - log_epsilons).mean()
        h_crvs = 0

    tc = sum(h_rv - h_crvs for h_rv in h_rvs) - (h_all - h_crvs)

    return tc / log_2


try:
    from sklearn.neighbors import KDTree

    total_correlation_ksg = _total_correlation_ksg_sklearn
except ImportError:
    total_correlation_ksg = _total_correlation_ksg_scipy


def _local_permutation(x, z, k_perm, rng):
    """
    Runge's local permutation: each sample takes the `x` of a distinct sample among
    its `k_perm` nearest neighbors in `z`, so the dependence of `x` on `z` survives.
    """
    n = len(x)
    if z.shape[1] == 0:
        return x[rng.permutation(n)]
    k_perm = min(k_perm, n)
    neighbors = cKDTree(z).query(z, k_perm, p=np.inf)[1].reshape(n, -1)
    used = np.zeros(n, dtype=bool)
    choice = np.empty(n, dtype=np.int64)
    for i in rng.permutation(n):
        candidates = neighbors[i][rng.permutation(neighbors.shape[1])]
        free = candidates[~used[candidates]]
        j = free[0] if len(free) else candidates[0]
        used[j] = True
        choice[i] = j
    return x[choice]


def conditional_mutual_information_test_knn(
    data, rvs, crvs=None, k=4, k_perm=5, n_surrogates=200, noise=1e-10, prng=None
):
    """
    Test :math:`I[X : Y \\mid Z] = 0` for continuous data with the KSG estimator
    and local-permutation surrogates :cite:`Runge2018`.

    Each surrogate replaces :math:`X` in sample :math:`i` by the :math:`X` of a
    (mostly distinct) sample among the `k_perm` nearest neighbors of :math:`i` in
    :math:`Z`. That keeps the dependence of :math:`X` on :math:`Z` while breaking
    any further dependence on :math:`Y`. It is the continuous analogue of
    :func:`~dit.inference.conditional_mutual_information_test`, which permutes
    within exact strata of a discrete :math:`Z`.

    Parameters
    ----------
    data : np.ndarray
        Samples, one per row.
    rvs : list of two lists
        The columns of :math:`X` and of :math:`Y`.
    crvs : list, None
        The columns of :math:`Z`. If None, :math:`X` is permuted freely.
    k : int
        Nearest neighbors for the KSG estimate.
    k_perm : int
        Neighborhood size for the local permutation; small values (5–10) keep the
        null conditional on :math:`Z`.
    n_surrogates : int
        The number of surrogates.
    noise : float
        Symmetry-breaking noise for the KSG estimator.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    result : SurrogateTest
    """
    from .significance import _result

    rng = as_generator(prng)
    data = np.asarray(data, dtype=np.float64)
    x_cols, y_cols = (list(r) for r in rvs)
    crvs = [] if crvs is None else list(crvs)
    value = total_correlation_ksg(data, [x_cols, y_cols], crvs, k=k, noise=noise, prng=rng)
    z = data[:, crvs]
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shuffled = data.copy()
        shuffled[:, x_cols] = _local_permutation(data[:, x_cols], z, k_perm, rng)
        null[i] = total_correlation_ksg(shuffled, [x_cols, y_cols], crvs, k=k, noise=noise, prng=rng)
    return _result(value, null, len(data))
