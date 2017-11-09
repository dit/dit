"""
Some estimators based on estimating local densities using k-nearest neighbors.
"""

from __future__ import division

import numpy as np

from scipy.special import digamma

from six.moves import zip

from dit.utils import flatten


__all__ = ['differential_entropy_knn',
           'total_correlation_ksg',
          ]

def _fuzz(data, noise):
    """
    Add noise to the data.

    Parameters
    ----------
    data : np.ndarray
        Data.
    noise : float
        The standard deviation of the normally-distributed noise to add to data.

    Returns
    -------
    data : np.ndarray
        The fuzzed data.
    """
    data = data.astype(np.float)
    data += np.random.normal(0.0, noise, size=data.shape)
    return data


def differential_entropy_knn(data, rvs=None, k=4, noise=1e-10):
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

    data = _fuzz(data, noise)

    d = len(rvs)

    tree = cKDTree(data[:, rvs])

    epsilons = tree.query(data[:, rvs], k + 1, p=np.inf)[0][:, -1]

    h = digamma(len(data)) - digamma(k) + d * (np.log(2) + np.log(epsilons).mean())

    return h / np.log(2)


def _total_correlation_ksg_scipy(data, rvs, crvs=None, k=4, noise=1e-10):
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

    Returns
    -------
    tc : float
        The total correlation of `rvs` given `crvs`.

    Notes
    -----
    The total correlation is computed in bits, not nats as most KSG estimators do.
    """
    # KSG suggest adding noise (to break symmetries?)
    data = _fuzz(data, noise)

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
        np.array([len(t.query_ball_point(point, epsilon, p=np.inf)) for point, epsilon in zip(data[:, rv], epsilons)])
        for rv, t in zip(rvs, tree_rvs)]

    log_epsilons = np.log(epsilons)

    h_rvs = [-digamma(n_rv).mean() for n_rv, d in zip(n_rvs, d_rvs)]

    h_all = -digamma(k)

    if crvs:
        tree_crvs = cKDTree(data[:, crvs])
        n_crvs = np.array([len(tree_crvs.query_ball_point(point, epsilon, p=np.inf)) for point, epsilon in
                           zip(data[:, crvs], epsilons)])
        h_crvs = -digamma(n_crvs).mean()
    else:
        h_rvs = [h_rv + digamma_N + d * (log_2 - log_epsilons).mean() for h_rv, d in zip(h_rvs, d_rvs)]
        h_all += digamma_N + sum(d_rvs) * (log_2 - log_epsilons).mean()
        h_crvs = 0

    tc = sum([h_rv - h_crvs for h_rv in h_rvs]) - (h_all - h_crvs)

    return tc / log_2


def _total_correlation_ksg_sklearn(data, rvs, crvs=None, k=4, noise=1e-10):
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
    data = _fuzz(data, noise)

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

    n_rvs = [t.query_radius(data[:, rv], epsilons, count_only=True) for rv, t in zip(rvs, tree_rvs)]

    log_epsilons = np.log(epsilons)

    h_rvs = [-digamma(n_rv).mean() for n_rv, d in zip(n_rvs, d_rvs)]

    h_all = -digamma(k)

    if crvs:
        tree_crvs = KDTree(data[:, crvs], metric="chebyshev")
        n_crvs = tree_crvs.query_radius(data[:, crvs], epsilons, count_only=True)
        h_crvs = -digamma(n_crvs).mean()
    else:
        h_rvs = [h_rv + digamma_N + d * (log_2 - log_epsilons).mean() for h_rv, d in zip(h_rvs, d_rvs)]
        h_all += digamma_N + sum(d_rvs) * (log_2 - log_epsilons).mean()
        h_crvs = 0

    tc = sum([h_rv - h_crvs for h_rv in h_rvs]) - (h_all - h_crvs)

    return tc / log_2


try:
    from sklearn.neighbors import KDTree
    total_correlation_ksg = _total_correlation_ksg_sklearn
except ImportError:
    from scipy.spatial import cKDTree
    total_correlation_ksg = _total_correlation_ksg_scipy
