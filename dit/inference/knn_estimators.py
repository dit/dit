"""
Some estimators based on estimating local densities using k-nearest neighbors.
"""

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[attr-defined]
from scipy.special import digamma, gammaln

from dit.utils import flatten

from ._symbols import as_generator

__all__ = (
    "cross_entropy_knn",
    "differential_entropy_knn",
    "jensen_shannon_divergence_knn",
    "kullback_leibler_divergence_knn",
    "mutual_information_mixed_ksg",
    "renyi_entropy_knn",
    "total_correlation_ksg",
    "tsallis_entropy_knn",
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


def _as_samples(data):
    data = np.asarray(data, dtype=np.float64)
    return data[:, None] if data.ndim == 1 else data


def cross_entropy_knn(x, y, k=4, noise=1e-10, prng=None):
    """
    Estimate the differential cross entropy :math:`-\\mathbb{E}_P[\\log q(X)]`
    from samples `x` of :math:`P` and `y` of :math:`Q` :cite:`Leonenko2008`.

    The density of :math:`Q` at each :math:`x_i` is estimated from the distance
    :math:`\\nu_k(i)` to its `k`-th nearest neighbor among the `y` (max norm):

    .. math::

        \\hat{H}(P, Q) = \\psi(m) - \\psi(k) + d \\log 2 + \\frac{d}{n} \\sum_i \\log \\nu_k(i)

    Parameters
    ----------
    x, y : np.ndarray
        Samples of :math:`P` (``n`` rows) and :math:`Q` (``m`` rows) with the
        same number ``d`` of columns.
    k : int
        The number of nearest neighbors.
    noise : float
        The standard deviation of the normally-distributed noise added to break ties.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    xh : float
        The estimated cross entropy, in bits.
    """
    rng = as_generator(prng)
    x, y = _fuzz(_as_samples(x), noise, rng), _fuzz(_as_samples(y), noise, rng)
    nu = cKDTree(y).query(x, k, p=np.inf)[0]
    nu = nu if nu.ndim == 1 else nu[:, -1]
    d = x.shape[1]
    h = digamma(len(y)) - digamma(k) + d * (np.log(2) + np.log(nu).mean())
    return h / np.log(2)


def kullback_leibler_divergence_knn(x, y, k=4, noise=1e-10, prng=None):
    """
    Estimate :math:`D_{KL}[P \\| Q]` from samples `x` of :math:`P` and `y` of :math:`Q`.

    This is the nearest-neighbor estimator of Wang, Kulkarni & Verdú
    :cite:`Wang2009`, :math:`\\frac{d}{n} \\sum_i \\log \\frac{\\nu_k(i)}{\\rho_k(i)}`
    plus a sample-size correction, where :math:`\\rho_k(i)` is the distance
    from :math:`x_i` to its `k`-th nearest neighbor among the other `x` and
    :math:`\\nu_k(i)` among the `y`. It equals :func:`cross_entropy_knn` minus
    :func:`differential_entropy_knn`.

    Parameters
    ----------
    x, y : np.ndarray
        Samples of :math:`P` and :math:`Q` with the same number of columns.
    k : int
        The number of nearest neighbors.
    noise : float
        The standard deviation of the normally-distributed noise added to break ties.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    dkl : float
        The estimated divergence, in bits.
    """
    rng = as_generator(prng)
    x, y = _fuzz(_as_samples(x), noise, rng), _fuzz(_as_samples(y), noise, rng)
    rho = cKDTree(x).query(x, k + 1, p=np.inf)[0][:, -1]
    nu = cKDTree(y).query(x, k, p=np.inf)[0]
    nu = nu if nu.ndim == 1 else nu[:, -1]
    d = x.shape[1]
    dkl = d * np.log(nu / rho).mean() + digamma(len(y)) - digamma(len(x))
    return dkl / np.log(2)


def jensen_shannon_divergence_knn(x, y, k=4, noise=1e-10, prng=None):
    """
    Estimate the Jensen–Shannon divergence of :math:`P` and :math:`Q` from samples.

    The pooled samples are a sample of the mixture
    :math:`M = \\pi P + (1 - \\pi) Q` with :math:`\\pi = n / (n + m)`, so this
    is :math:`h(M) - \\pi h(P) - (1 - \\pi) h(Q)` with each differential entropy
    from :func:`differential_entropy_knn` :cite:`Lin1991`. With equal sample
    sizes it is the usual (equally weighted) divergence.

    Parameters
    ----------
    x, y : np.ndarray
        Samples of :math:`P` and :math:`Q` with the same number of columns.
    k : int
        The number of nearest neighbors.
    noise : float
        The standard deviation of the normally-distributed noise added to break ties.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    jsd : float
        The estimated divergence, in bits.
    """
    rng = as_generator(prng)
    x, y = _as_samples(x), _as_samples(y)
    pi = len(x) / (len(x) + len(y))
    h = lambda data: differential_entropy_knn(data, k=k, noise=noise, prng=rng)  # noqa: E731
    return h(np.concatenate([x, y])) - pi * h(x) - (1 - pi) * h(y)


def _renyi_information(data, order, k, noise, prng):
    """
    The Leonenko–Pronzato–Savani estimate of :math:`\\int f^q`.
    """
    if not order < k + 1:
        raise ValueError("The estimator needs `order` < `k` + 1.")
    data = _fuzz(_as_samples(data), noise, prng)
    N, d = data.shape
    rho = cKDTree(data).query(data, k + 1, p=np.inf)[0][:, -1]
    # zeta_i = (N - 1) C_k V_d rho_i^d with C_k = (Gamma(k) / Gamma(k + 1 - q))^(1 / (1 - q)), V_d = 2^d
    log_zeta = np.log(N - 1) + (gammaln(k) - gammaln(k + 1 - order)) / (1 - order) + d * np.log(2) + d * np.log(rho)
    return np.exp((1 - order) * log_zeta).mean()


def renyi_entropy_knn(data, order, k=4, noise=1e-10, prng=None):
    """
    Estimate the differential Rényi entropy
    :math:`\\frac{1}{1 - q} \\log \\int f^q` from samples :cite:`Leonenko2008`.

    Parameters
    ----------
    data : np.ndarray
        The samples, one per row.
    order : float
        The order :math:`q`; it must be less than ``k + 1``. Order 1 gives
        :func:`differential_entropy_knn`.
    k : int
        The number of nearest neighbors.
    noise : float
        The standard deviation of the normally-distributed noise added to break ties.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    h : float
        The estimated Rényi entropy, in bits.
    """
    if order == 1:
        return differential_entropy_knn(_as_samples(data), k=k, noise=noise, prng=prng)
    return np.log2(_renyi_information(data, order, k, noise, prng)) / (1 - order)


def tsallis_entropy_knn(data, order, k=4, noise=1e-10, prng=None):
    """
    Estimate the differential Tsallis entropy
    :math:`\\frac{1}{q - 1} \\left(1 - \\int f^q\\right)` from samples :cite:`Leonenko2008`.

    Parameters
    ----------
    data : np.ndarray
        The samples, one per row.
    order : float
        The order :math:`q`; it must be less than ``k + 1``. Order 1 gives the
        Shannon differential entropy in nats.
    k : int
        The number of nearest neighbors.
    noise : float
        The standard deviation of the normally-distributed noise added to break ties.
    prng : None, int, Generator, RandomState
        Source of randomness for the noise.

    Returns
    -------
    s : float
        The estimated Tsallis entropy.
    """
    if order == 1:
        return differential_entropy_knn(_as_samples(data), k=k, noise=noise, prng=prng) * np.log(2)
    return (1 - _renyi_information(data, order, k, noise, prng)) / (order - 1)


def mutual_information_mixed_ksg(x, y, k=5):
    """
    Estimate :math:`I[X : Y]` from samples that may be discrete, continuous, or
    a mixture of the two :cite:`Gao2017`.

    This is the mixed KSG estimator: with :math:`\\rho_i` the max-norm distance
    from sample :math:`i` to its `k`-th nearest neighbor in the joint space,

    .. math::

        \\hat{I} = \\frac{1}{N} \\sum_i \\left[\\psi(\\tilde{k}_i) + \\log N - \\psi(n_{x,i}) - \\psi(n_{y,i})\\right]

    where :math:`n_{x,i}` and :math:`n_{y,i}` count the samples (including
    :math:`i`) strictly within :math:`\\rho_i` in each marginal, and
    :math:`\\tilde{k}_i = k` unless :math:`\\rho_i = 0`, in which case
    :math:`\\tilde{k}_i`, :math:`n_{x,i}` and :math:`n_{y,i}` count the exact
    repeats of sample :math:`i`. No noise is added, since repeated values are
    what identify the discrete part.

    Parameters
    ----------
    x, y : np.ndarray
        Paired samples, one per row.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    mi : float
        The estimated mutual information, in bits.
    """
    x, y = _as_samples(x), _as_samples(y)
    if len(x) != len(y):
        raise ValueError("`x` and `y` must have the same number of samples.")
    if not k < len(x):
        raise ValueError("`k` must be smaller than the number of samples.")
    N = len(x)
    xy = np.hstack([x, y])
    rho = cKDTree(xy).query(xy, k + 1, p=np.inf)[0][:, -1]
    ties = rho == 0
    radius = np.where(ties, 1e-15, rho - 1e-15)
    n_x = cKDTree(x).query_ball_point(x, radius, p=np.inf, return_length=True)
    n_y = cKDTree(y).query_ball_point(y, radius, p=np.inf, return_length=True)
    k_tilde = np.full(N, float(k))
    if ties.any():
        k_tilde[ties] = cKDTree(xy).query_ball_point(xy[ties], 1e-15, p=np.inf, return_length=True)
    mi = np.mean(digamma(k_tilde) + np.log(N) - digamma(n_x) - digamma(n_y))
    return mi / np.log(2)


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
    # Marginal counts are of points strictly closer than epsilon (Kraskov et al.; Frenzel
    # and Pompe); the k-th neighbor itself lies on the boundary in some subspace.
    radii = np.nextafter(epsilons, 0)

    n_rvs = [
        t.query_ball_point(data[:, rv], radii, p=np.inf, return_length=True)
        for rv, t in zip(rvs, tree_rvs, strict=True)
    ]

    log_epsilons = np.log(epsilons)

    h_rvs = [-digamma(n_rv).mean() for n_rv, d in zip(n_rvs, d_rvs, strict=True)]

    h_all = -digamma(k)

    if crvs:
        tree_crvs = cKDTree(data[:, crvs])
        n_crvs = tree_crvs.query_ball_point(data[:, crvs], radii, p=np.inf, return_length=True)
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
    # Marginal counts are of points strictly closer than epsilon (Kraskov et al.; Frenzel
    # and Pompe); the k-th neighbor itself lies on the boundary in some subspace.
    radii = np.nextafter(epsilons, 0)

    n_rvs = [t.query_radius(data[:, rv], radii, count_only=True) for rv, t in zip(rvs, tree_rvs, strict=True)]

    log_epsilons = np.log(epsilons)

    h_rvs = [-digamma(n_rv).mean() for n_rv, d in zip(n_rvs, d_rvs, strict=True)]

    h_all = -digamma(k)

    if crvs:
        tree_crvs = KDTree(data[:, crvs], metric="chebyshev")
        n_crvs = tree_crvs.query_radius(data[:, crvs], radii, count_only=True)
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
