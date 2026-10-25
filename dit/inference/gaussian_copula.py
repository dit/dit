"""
Gaussian-copula estimates of mutual information.
"""

import numpy as np
from scipy.special import digamma, ndtri

__all__ = (
    "copula_normalize",
    "gaussian_copula_mutual_information",
)


def copula_normalize(data):
    """
    Map each column of `data` through its empirical CDF and then the inverse
    standard normal CDF, giving standard normal marginals with the same ranks.

    Parameters
    ----------
    data : array_like
        The samples, one per row.

    Returns
    -------
    normed : np.ndarray
        The copula-normalized samples, as a 2D array.
    """
    data = np.asarray(data, dtype=float)
    data = data[:, None] if data.ndim == 1 else data
    ranks = np.argsort(np.argsort(data, axis=0, kind="mergesort"), axis=0, kind="mergesort")
    return ndtri((ranks + 1.0) / (len(data) + 1.0))


def _gaussian_entropy(data, bias_correct):
    """
    The entropy (nats) of a Gaussian with the sample covariance of `data`,
    less the analytic bias of its log-determinant.
    """
    N, d = data.shape
    if d == 0:
        return 0.0
    if d >= N:
        raise ValueError("More samples than variables are needed.")
    centered = data - data.mean(axis=0)
    chol = np.linalg.cholesky(centered.T @ centered / (N - 1))
    h = np.log(np.diag(chol)).sum() + 0.5 * d * (np.log(2 * np.pi) + 1)
    if bias_correct:
        h -= d * (np.log(2) - np.log(N - 1)) / 2 + np.sum(digamma((N - np.arange(1, d + 1)) / 2) / 2)
    return h


def gaussian_copula_mutual_information(x, y, z=None, bias_correct=True):
    """
    Estimate :math:`I[X : Y \\mid Z]` with the Gaussian copula :cite:`Ince2017`.

    Each variable is copula-normalized (:func:`copula_normalize`) and the
    information is computed for a Gaussian with the resulting covariance. This
    is a lower bound on the true information (exact when the copula is
    Gaussian), insensitive to the marginals, and fast and stable in high
    dimensions.

    Parameters
    ----------
    x, y : array_like
        Paired continuous samples, one per row.
    z : array_like, None
        Samples of the conditioning variable. If None, estimate :math:`I[X : Y]`.
    bias_correct : bool
        Remove the analytic finite-sample bias of each Gaussian entropy.

    Returns
    -------
    mi : float
        The estimate, in bits.
    """
    parts = [copula_normalize(v) for v in (x, y)]
    if len({len(p) for p in parts}) != 1:
        raise ValueError("`x` and `y` must have the same number of samples.")
    cz = np.empty((len(parts[0]), 0)) if z is None else copula_normalize(z)
    cx, cy = parts

    def H(*blocks):
        return _gaussian_entropy(np.hstack(blocks), bias_correct)

    return float((H(cx, cz) + H(cy, cz) - H(cx, cy, cz) - H(cz)) / np.log(2))
