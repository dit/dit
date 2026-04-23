"""
Statistics for distributions with numerical outcomes.

All functions operate on finite, discrete distributions and raise
``TypeError`` if the outcomes are not numeric.
"""

import numpy as np

from ..helpers import numerical_test

__all__ = (
    "cdf",
    "central_moment",
    "correlation",
    "covariance",
    "expectation",
    "iqr",
    "kurtosis",
    "maximum",
    "mean",
    "median",
    "minimum",
    "mode",
    "percentile",
    "quantile",
    "range_",
    "skewness",
    "standard_deviation",
    "standard_moment",
    "variance",
)


# ── Core moments ──────────────────────────────────────────────────────────


def _scalar(val):
    """Convert 0-d numpy scalars to plain Python float."""
    if isinstance(val, np.generic) and val.ndim == 0:
        return float(val)
    return val


def expectation(dist, f=None):
    """
    Compute E[f(X)] for an arbitrary callable *f*.

    When *f* is ``None`` this is equivalent to :func:`mean`.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    f : callable or None
        A function that maps an outcome to a numeric value (scalar or array).
        If ``None``, the identity function is used.

    Returns
    -------
    result : float or ndarray
        The expected value E[f(X)].
    """
    numerical_test(dist)
    outcomes, pmf = zip(*dist.zipped(mode="patoms"), strict=True)
    pmf = np.asarray(pmf)
    vals = np.asarray(outcomes) if f is None else np.asarray([f(o) for o in outcomes])
    return _scalar(np.average(vals, axis=0, weights=pmf))


def mean(dist):
    """
    Compute the mean (expected value) of a distribution.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    means : ndarray
        The mean of each index of the outcomes.
    """
    return expectation(dist)


def central_moment(dist, n):
    """
    Compute the *n*-th central moment E[(X - mu)^n].

    Parameters
    ----------
    dist : Distribution
    n : int
        Which moment to compute.

    Returns
    -------
    moments : ndarray
        The *n*-th central moment of each index of the outcomes.
    """
    mu = mean(dist)
    outcomes, pmf = zip(*dist.zipped(mode="patoms"), strict=True)
    outcomes = np.asarray(outcomes)
    pmf = np.asarray(pmf)
    terms = np.asarray([(np.asarray(o) - mu) ** n for o in outcomes])
    terms[np.isnan(terms)] = 0
    return _scalar(np.average(terms, axis=0, weights=pmf))


def standard_moment(dist, n):
    """
    Compute the *n*-th standardised moment E[((X - mu)/sigma)^n].

    Parameters
    ----------
    dist : Distribution
    n : int

    Returns
    -------
    moments : ndarray
    """
    return central_moment(dist, n) / standard_deviation(dist) ** n


def variance(dist):
    """
    Compute the variance Var(X) = E[(X - mu)^2].

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    var : ndarray
        The variance of each index of the outcomes.
    """
    return central_moment(dist, 2)


def standard_deviation(dist):
    """
    Compute the standard deviation sqrt(Var(X)).

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    std : ndarray
    """
    return _scalar(np.sqrt(variance(dist)))


def skewness(dist):
    """
    Compute the skewness (3rd standardised moment).

    Measures asymmetry of the distribution.  Zero for symmetric
    distributions, positive for right-skewed, negative for left-skewed.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    skew : ndarray
    """
    return standard_moment(dist, 3)


def kurtosis(dist, excess=True):
    """
    Compute the kurtosis (4th standardised moment).

    Parameters
    ----------
    dist : Distribution
    excess : bool
        If ``True`` (default), return *excess* kurtosis (subtract 3 so
        that a normal distribution has kurtosis 0).  If ``False``,
        return the raw 4th standardised moment.

    Returns
    -------
    kurt : ndarray
    """
    k = standard_moment(dist, 4)
    if excess:
        k = k - 3
    return k


# ── Joint-distribution statistics ────────────────────────────────────────


def covariance(dist):
    """
    Compute the covariance matrix for a joint distribution.

    For a distribution over *d* random variables, returns a *d x d*
    matrix where entry (i, j) is Cov(Xi, Xj).

    For a 1-D distribution the result is a 1x1 array containing the
    variance.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    cov : ndarray, shape (d, d)
    """
    numerical_test(dist)
    outcomes, pmf = zip(*dist.zipped(mode="patoms"), strict=True)
    outcomes = np.asarray(outcomes, dtype=float)
    pmf = np.asarray(pmf)

    if outcomes.ndim == 1:
        outcomes = outcomes[:, np.newaxis]

    mu = np.average(outcomes, axis=0, weights=pmf)
    centered = outcomes - mu
    return np.einsum("i,ij,ik->jk", pmf, centered, centered)


def correlation(dist):
    """
    Compute the Pearson correlation matrix for a joint distribution.

    For a distribution over *d* random variables, returns a *d x d*
    matrix where entry (i, j) is Cor(Xi, Xj) = Cov(Xi, Xj) / (sigma_i * sigma_j).

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    cor : ndarray, shape (d, d)
    """
    cov = covariance(dist)
    std = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore"):
        cor = cov / np.outer(std, std)
    cor[np.isnan(cor)] = 0.0
    np.fill_diagonal(cor, 1.0)
    return cor


# ── Quantile / order statistics ──────────────────────────────────────────


def quantile(dist, q):
    """
    Compute the *q*-quantile of a distribution (0 <= q <= 1).

    Uses the "lower" interpolation method: returns the largest outcome
    *x* such that P(X <= x) <= q, averaging with the next outcome when
    P(X <= x) == q exactly (matching :func:`median` behaviour).

    For multi-dimensional distributions, operates on the marginal of
    each index independently.

    Parameters
    ----------
    dist : Distribution
    q : float
        Probability level in [0, 1].

    Returns
    -------
    quantiles : ndarray
    """
    numerical_test(dist)
    cum = dist.pmf.cumsum()
    outcomes = np.asarray(dist.outcomes)

    if outcomes.ndim == 1:
        outcomes = outcomes[:, np.newaxis]

    d = outcomes.shape[1]
    result = np.empty(d)

    for idx in range(d):
        vals = outcomes[:, idx]
        order = np.argsort(vals)
        sorted_vals = vals[order]
        sorted_cum = cum[order]

        mask_gt = sorted_cum > q
        mask_ge = sorted_cum >= q

        g = sorted_vals[-1] if not mask_gt.any() else sorted_vals[mask_gt.argmax()]
        ge = sorted_vals[-1] if not mask_ge.any() else sorted_vals[mask_ge.argmax()]

        result[idx] = (g + ge) / 2

    if d == 1:
        return float(result[0])
    return result


def percentile(dist, p):
    """
    Compute the *p*-th percentile (0 <= p <= 100).

    Equivalent to ``quantile(dist, p / 100)``.

    Parameters
    ----------
    dist : Distribution
    p : float
        Percentile in [0, 100].

    Returns
    -------
    result : ndarray
    """
    return quantile(dist, p / 100)


def median(dist):
    """
    Compute the median of a distribution.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    medians : ndarray
        The median of each index of the outcomes.
    """
    return quantile(dist, 0.5)


def iqr(dist):
    """
    Compute the interquartile range (Q3 - Q1).

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    iqr : ndarray
    """
    return quantile(dist, 0.75) - quantile(dist, 0.25)


def minimum(dist):
    """
    Return the smallest outcome in the support (per index).

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    mins : ndarray
    """
    numerical_test(dist)
    outcomes = np.asarray(dist.outcomes)
    if outcomes.ndim == 1:
        return outcomes.min()
    return outcomes.min(axis=0)


def maximum(dist):
    """
    Return the largest outcome in the support (per index).

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    maxs : ndarray
    """
    numerical_test(dist)
    outcomes = np.asarray(dist.outcomes)
    if outcomes.ndim == 1:
        return outcomes.max()
    return outcomes.max(axis=0)


def range_(dist):
    """
    Compute the range (maximum - minimum) of the support per index.

    (Named ``range_`` to avoid shadowing the built-in ``range``.)

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    r : ndarray
    """
    return maximum(dist) - minimum(dist)


# ── CDF ──────────────────────────────────────────────────────────────────


def cdf(dist):
    """
    Compute the cumulative distribution function P(X <= x).

    For a 1-D distribution, returns ``(values, cumprobs)`` where
    *values* are the sorted unique outcomes and *cumprobs* are
    the corresponding cumulative probabilities.

    For a multi-dimensional distribution, returns a list of
    ``(values, cumprobs)`` tuples, one per index (marginal CDF).

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    result : tuple or list of tuples
        For 1-D: ``(values, cumprobs)``.
        For multi-D: ``[(values_0, cumprobs_0), …]``.
    """
    numerical_test(dist)
    outcomes, pmf = zip(*dist.zipped(mode="patoms"), strict=True)
    outcomes = np.asarray(outcomes, dtype=float)
    pmf = np.asarray(pmf)

    if outcomes.ndim == 1:
        outcomes = outcomes[:, np.newaxis]

    d = outcomes.shape[1]
    results = []

    for idx in range(d):
        vals = outcomes[:, idx]
        unique_vals = np.unique(vals)
        cumprobs = np.empty_like(unique_vals)
        for i, v in enumerate(unique_vals):
            cumprobs[i] = pmf[vals <= v].sum()
        results.append((unique_vals, cumprobs))

    if d == 1:
        return results[0]
    return results


# ── Existing functions (kept for backwards compat) ───────────────────────


def mode(dist):
    """
    Compute the modes of a distribution.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    modes : list of ndarray
        A list of arrays, one for each index of the outcomes. Each array
        contains the modes of that index.
    """
    numerical_test(dist)

    try:
        dists = [dist.marginal([i]) for i in range(dist.outcome_length())]
    except AttributeError:
        dists = [dist]

    modes = [np.asarray(d.outcomes)[d.pmf == d.pmf.max()] for d in dists]
    modes = [m.flatten() for m in modes]
    return modes
