"""
Tests and selection criteria for the Markov order of a sequence.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ._symbols import Trials, as_generator, check_sampling, standardize_trials, word_codes
from .surrogates import _whittle_trials

__all__ = (
    "MarkovOrderTest",
    "markov_order_test",
    "select_markov_order",
)


@dataclass(frozen=True)
class MarkovOrderTest:
    """
    The result of :func:`markov_order_test`.

    Attributes
    ----------
    order : int
        The Markov order of the null hypothesis.
    statistic : str
        The name of the test statistic.
    value : float
        The observed value of the statistic.
    pvalue : float
        The p-value of the observed statistic.
    dof : int
        Degrees of freedom of the asymptotic chi-squared distribution.
    null : np.ndarray or None
        The statistic evaluated on each surrogate (exact tests only).
    n_windows : int or None
        The number of ``(order + 2)``-windows (the effective sample size).
    """

    order: int
    statistic: str
    value: float
    pvalue: float
    dof: int
    null: np.ndarray | None = None
    n_windows: int | None = None


def _contingency(trials, order, M):
    """
    For each ``order``-word ``u``, the table of counts of ``a u b`` with rows
    ``a`` and columns ``b``, flattened as parallel arrays over observed words.
    `trials` is a list of code arrays; windows never span two trials.
    """
    trials = [np.asarray(c, dtype=np.int64) for c in trials if len(c) >= order + 2]
    windows = np.concatenate([np.lib.stride_tricks.sliding_window_view(c, order + 2) for c in trials])
    a = windows[:, 0]
    b = windows[:, -1]
    u = word_codes([c[1:-1] for c in trials], order, M) if order else np.zeros(len(windows), dtype=np.int64)
    _, u = np.unique(u, return_inverse=True)
    u = u.ravel()
    L = int(u.max()) + 1
    keys = (a * L + u) * M + b
    uniq, O = np.unique(keys, return_counts=True)
    b = uniq % M
    u = (uniq // M) % L
    a = uniq // (M * L)
    return a, u, b, O.astype(float)


def _group_sums(keys, values):
    uniq, inverse = np.unique(keys, return_inverse=True)
    return np.bincount(inverse.ravel(), weights=values)[inverse.ravel()], len(uniq)


def _statistics(trials, order, M):
    """
    The plug-in conditional entropies ``h_{order+1}`` and ``h_order`` (bits,
    on common targets), Pearson's chi-squared statistic, the degrees of
    freedom, and the number of ``(order + 2)``-grams.
    """
    a, u, b, O = _contingency(trials, order, M)
    total = O.sum()
    U = int(u.max()) + 1
    C_au, _ = _group_sums(a * U + u, O)
    C_ub, _ = _group_sums(u * M + b, O)
    C_u, _ = _group_sums(u, O)

    h_long = float(-np.sum(O * np.log2(O / C_au)) / total)
    h_short = float(-np.sum(O * np.log2(C_ub / C_u)) / total)

    # Summed over every cell with C(au) C(ub) > 0 (Anderson & Goodman), the
    # expected counts total the observed counts, so
    # sum (O - E)^2 / E = sum O^2 / E - total.
    chi_sq = float(np.sum(O**2 * C_u / (C_au * C_ub)) - total)

    rows = np.bincount(np.unique(u * M + a) // M, minlength=U)
    cols = np.bincount(np.unique(u * M + b) // M, minlength=U)
    dof = int(np.sum(np.maximum(rows - 1, 0) * np.maximum(cols - 1, 0)))
    return h_long, h_short, chi_sq, dof, total


def markov_order_test(
    data,
    order=1,
    statistic="entropy_rate",
    method="exact",
    n_surrogates=1000,
    prng=None,
):
    """
    Test the null hypothesis that `data` is Markov of order `order` against the
    alternative that it is of order ``order + 1``.

    Parameters
    ----------
    data : iterable or Trials
        The observed sequence, or independent :class:`~dit.inference.Trials`.
        Rows of a 2D array are joint symbols.
    order : int
        The Markov order of the null hypothesis.
    statistic : {'entropy_rate', 'chi2'}
        ``'entropy_rate'`` uses the plug-in conditional entropy
        :math:`H[X_{n+1} \\mid X_{0:n+1}]` with ``n = order``; small values are
        evidence against the null. ``'chi2'`` uses Pearson's statistic for the
        ``(order + 2)``-gram counts; large values are evidence against the null.
    method : {'exact', 'asymptotic'}
        ``'exact'`` compares against :func:`whittle_surrogates`, which is valid
        at any sample size :cite:`Pethel2014`. ``'asymptotic'`` uses the
        chi-squared limit :cite:`Anderson1957`; for ``'entropy_rate'`` this is
        the G-test, since
        :math:`G = 2 N \\ln 2 \\, (h_n - h_{n+1})`.
    n_surrogates : int
        The number of surrogates for the exact test.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    result : MarkovOrderTest

    Notes
    -----
    Pethel & Hahs show that the asymptotic test is badly anti-conservative for
    short sequences and higher orders (e.g. a size of 0.22 instead of 0.05 for
    third-order chains on four symbols with 400 samples), while the exact test
    keeps its nominal size :cite:`Pethel2014`. The exact p-value is
    ``(1 + #{surrogates at least as extreme}) / (1 + n_surrogates)``.
    An :class:`~dit.inference.UndersamplingWarning` is raised when the
    ``(order + 2)``-gram counts are sparse; the exact test stays valid, but its
    power is low.

    For :class:`~dit.inference.Trials`, counts are pooled over trials and each
    surrogate draws every trial independently from its own surrogate set, which
    is the exact conditional null given all trials' counts.
    """
    return _markov_order_test(data, order, statistic, method, n_surrogates, prng, check=True)


def _markov_order_test(data, order, statistic, method, n_surrogates, prng, check):
    if statistic not in ("entropy_rate", "chi2"):
        raise ValueError(f"Unknown statistic {statistic!r}.")
    if method not in ("exact", "asymptotic"):
        raise ValueError(f"Unknown method {method!r}.")
    trials, alphabet = standardize_trials(data)
    M = len(alphabet)
    if all(len(c) < order + 2 for c in trials):
        raise ValueError("`data` is too short to test this order.")
    h, h_n, chi_sq, dof, total = _statistics(trials, order, M)
    value = h if statistic == "entropy_rate" else chi_sq
    if check:
        distinct = len(_contingency(trials, order, M)[3])
        check_sampling(distinct, int(total), M, order + 2, stacklevel=4)

    if method == "asymptotic":
        if statistic == "chi2":
            pvalue = chi2.sf(chi_sq, dof) if dof > 0 else 1.0
        else:
            g = 2 * total * np.log(2) * max(h_n - h, 0.0)
            pvalue = chi2.sf(g, dof) if dof > 0 else 1.0
        return MarkovOrderTest(order, statistic, value, float(pvalue), dof, None, int(total))

    rng = as_generator(prng)
    surrogates = _whittle_trials(trials, order, n_surrogates, rng)
    null = np.empty(n_surrogates)
    for i, s in enumerate(surrogates):
        h_s, _, chi_s, _, _ = _statistics(s, order, M)
        null[i] = h_s if statistic == "entropy_rate" else chi_s
    tol = 1e-12 * max(1.0, abs(value))
    extreme = np.sum(null <= value + tol) if statistic == "entropy_rate" else np.sum(null >= value - tol)
    pvalue = (1 + extreme) / (1 + n_surrogates)
    return MarkovOrderTest(order, statistic, value, float(pvalue), dof, null, int(total))


def _log_likelihoods(trials, max_order, M):
    """
    Maximized log-likelihoods (nats) of orders ``0..max_order``, all evaluated
    on the targets ``x_t`` for ``t >= max_order`` of each trial.
    """
    lls = []
    for k in range(max_order + 1):
        segments = [c[max_order - k :] for c in trials if len(c) > max_order]
        joint = word_codes(segments, k + 1, M)
        past = word_codes([seg[:-1] for seg in segments], k, M)
        _, jc = np.unique(joint, return_counts=True)
        _, pc = np.unique(past, return_counts=True)
        lls.append(float((jc * np.log(jc)).sum() - (pc * np.log(pc)).sum()))
    return np.array(lls)


def select_markov_order(
    data,
    max_order,
    method="exact",
    alpha=0.05,
    n_surrogates=1000,
    prng=None,
):
    """
    Estimate the Markov order of `data`.

    Parameters
    ----------
    data : iterable or Trials
        The observed sequence, or independent :class:`~dit.inference.Trials`.
        Rows of a 2D array are joint symbols.
    max_order : int
        The largest order considered.
    method : {'exact', 'chi2', 'aic', 'bic'}
        ``'exact'`` and ``'chi2'`` test order ``n`` against ``n + 1`` for
        ``n = 0, 1, ...`` and return the first order that is not rejected at
        level `alpha` (``'exact'`` uses :func:`markov_order_test` with Whittle
        surrogates :cite:`Pethel2014`; ``'chi2'`` its asymptotic G-test).
        ``'aic'`` and ``'bic'`` minimize the information criterion over orders
        fitted on a common set of targets :cite:`Tong1975,Katz1981`.
    alpha : float
        The significance level for the testing methods.
    n_surrogates : int
        The number of surrogates per test for ``'exact'``.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    order : int
        The estimated order.

    Notes
    -----
    An :class:`~dit.inference.UndersamplingWarning` is raised when
    ``max_order + 1``-words are sparse. The selected order then reflects what the
    data can resolve, and may be smaller than the process's true order.

    A process with infinite Markov order (e.g. a strictly sofic process such as
    the even process) has no true order; every method returns larger orders as
    the sample grows.
    """
    trials, alphabet = standardize_trials(data)
    M = len(alphabet)
    max_order = min(int(max_order), max(max(len(c) for c in trials) - 2, 0))
    top = word_codes(trials, max_order + 1, M)
    check_sampling(len(np.unique(top)), len(top), M, max_order + 1)
    if method in ("exact", "chi2"):
        rng = as_generator(prng)
        for n in range(max_order):
            result = _markov_order_test(
                Trials(trials),
                n,
                "entropy_rate",
                "exact" if method == "exact" else "asymptotic",
                n_surrogates,
                rng,
                check=False,
            )
            if result.pvalue > alpha:
                return n
        return max_order
    if method in ("aic", "bic"):
        lls = _log_likelihoods(trials, max_order, M)
        params = np.array([M**k * (M - 1) for k in range(max_order + 1)], dtype=float)
        targets = sum(max(len(c) - max_order, 0) for c in trials)
        penalty = 2.0 if method == "aic" else np.log(targets)
        scores = -2 * lls + penalty * params
        return int(np.argmin(scores))
    raise ValueError(f"Unknown method {method!r}.")
