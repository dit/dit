"""
Sample estimates of conditional mutual information and transfer entropy, with
surrogate-based significance tests and bootstrap confidence intervals.
"""

from dataclasses import dataclass

import numpy as np

from ._symbols import Trials, as_generator, check_sampling, is_trials, standardize_trials
from .estimators import _entropy
from .surrogates import block_surrogates, shift_surrogates, whittle_surrogates

__all__ = (
    "SurrogateTest",
    "benjamini_hochberg",
    "bootstrap_ci",
    "conditional_mutual_information",
    "conditional_mutual_information_test",
    "stationary_bootstrap",
    "transfer_entropy",
    "transfer_entropy_ci",
    "transfer_entropy_test",
)


@dataclass(frozen=True)
class SurrogateTest:
    """
    The result of a surrogate significance test.

    Attributes
    ----------
    value : float
        The observed statistic.
    pvalue : float
        ``(1 + #{null >= value}) / (1 + len(null))``.
    null : np.ndarray
        The statistic evaluated on each surrogate.
    n_samples : int or None
        The number of aligned samples (windows) the statistic was computed from.
    """

    value: float
    pvalue: float
    null: np.ndarray
    n_samples: int | None = None


def _dense(values):
    """
    Integer codes ``0..K-1`` for a 1D array of labels, or for the rows of a 2D array.
    """
    values = np.asarray(values)
    if values.ndim > 1:
        _, inverse = np.unique(values, axis=0, return_inverse=True)
    else:
        _, inverse = np.unique(values, return_inverse=True)
    return inverse.ravel().astype(np.int64)


def _pair(a, b):
    return _dense(a * (int(b.max()) + 1) + b)


def _cmi_codes(x, y, z, estimator):
    """
    :math:`I[x : y \\mid z]` in bits from dense integer codes.
    """

    def H(codes):
        return _entropy(np.bincount(codes), estimator)

    xz, yz = _pair(x, z), _pair(y, z)
    xyz = _pair(xz, y)
    return H(xz) + H(yz) - H(xyz) - H(z)


def conditional_mutual_information(x, y, z=None, estimator="plugin"):
    """
    Estimate :math:`I[X : Y \\mid Z]` in bits from paired samples.

    Parameters
    ----------
    x, y : array_like
        Samples of each variable; rows of 2D arrays are joint outcomes.
    z : array_like, None
        Samples of the conditioning variable. If None, estimate :math:`I[X : Y]`.
    estimator : str
        The entropy estimator applied to each term; see
        :func:`~dit.inference.entropy_from_counts`.

    Returns
    -------
    cmi : float
    """
    x, y = _dense(x), _dense(y)
    z = np.zeros_like(x) if z is None else _dense(z)
    _check_joint(x, y, z)
    return _cmi_codes(x, y, z, estimator)


def _within_strata(x, z, rng):
    """
    Permute `x` uniformly within each stratum of `z`.
    """
    grouped = np.argsort(z, kind="stable")
    shuffled = np.lexsort((rng.random(len(z)), z))
    out = np.empty_like(x)
    out[grouped] = x[shuffled]
    return out


def conditional_mutual_information_test(x, y, z=None, n_surrogates=1000, estimator="plugin", prng=None):
    """
    Test :math:`I[X : Y \\mid Z] = 0` by permuting `x` within strata of `z`.

    Parameters
    ----------
    x, y : array_like
        Samples of each variable; rows of 2D arrays are joint outcomes.
    z : array_like, None
        Samples of the conditioning variable. If None, `x` is permuted freely.
    n_surrogates : int
        The number of permutations.
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    result : SurrogateTest

    Notes
    -----
    Permuting within strata preserves the joint distributions of :math:`(X, Z)`
    and :math:`(Y, Z)` while breaking any dependence between :math:`X` and
    :math:`Y` given :math:`Z`. The test is exact for exchangeable samples; for
    serially dependent samples it is approximate.
    """
    rng = as_generator(prng)
    x, y = _dense(x), _dense(y)
    z = np.zeros_like(x) if z is None else _dense(z)
    _check_joint(x, y, z)
    value = _cmi_codes(x, y, z, estimator)
    null = np.array([_cmi_codes(_within_strata(x, z, rng), y, z, estimator) for _ in range(n_surrogates)])
    return _result(value, null, len(x))


def _result(value, null, n_samples=None):
    tol = 1e-12 * max(1.0, abs(value))
    pvalue = float((1 + np.sum(null >= value - tol)) / (1 + len(null)))
    return SurrogateTest(float(value), pvalue, null, n_samples)


def _check_joint(*codes):
    """
    Warn if the joint outcomes of aligned code arrays are undersampled.
    """
    joint = codes[0]
    for c in codes[1:]:
        joint = _pair(joint, c)
    check_sampling(len(np.unique(joint)), len(joint), stacklevel=4)


def _window_codes(codes, L, start, stop):
    """
    Rows ``codes[s:s + L]`` for ``s`` in ``range(start, stop)``.
    """
    if L == 0:
        return np.zeros((stop - start, 0), dtype=np.int64)
    return np.lib.stride_tricks.sliding_window_view(codes, L)[start:stop]


def _dense_rows(rows):
    if rows.shape[1] == 0:
        return np.zeros(len(rows), dtype=np.int64)
    return _dense(rows)


def _resolve_te(target, history_length, source_history, lag, max_history, prng):
    """
    Validate transfer entropy parameters, resolving ``history_length='auto'``.
    """
    if isinstance(history_length, str):
        if history_length != "auto":
            raise ValueError(f"Unknown history_length {history_length!r}.")
        from .markov_order import select_markov_order

        trials, alphabet = standardize_trials(target)
        if max_history is None:
            N = sum(len(t) for t in trials)
            M = max(len(alphabet), 2)
            max_history = max(int(np.log(max(N, 1) / 5) / np.log(M)) - 1, 1)
        history_length = select_markov_order(target, max_history, prng=prng)
    k = int(history_length)
    l = max(k, 1) if source_history is None else int(source_history)
    if k < 0:
        raise ValueError("`history_length` must be non-negative.")
    if l < 1:
        raise ValueError("`source_history` must be at least 1.")
    if lag < 1:
        raise ValueError("`lag` must be at least 1.")
    return k, l, int(lag)


def _te_codes(source, target, k, l=None, lag=1, conditions=()):
    """
    Aligned (source past, target present, context) codes, pooled over trials.

    For each time ``t`` the source past is ``X[t - lag - l + 1 : t - lag + 1]``,
    the target past ``Y[t - k : t]``, and the context joins the target past with
    each condition's past ``Z[t - m : t]``, ``m = max(k, 1)``.
    """
    l = max(k, 1) if l is None else l
    series = [source, target, *conditions]
    kinds = {is_trials(x) for x in series}
    if len(kinds) > 1:
        raise ValueError("`source`, `target` and `conditions` must all be Trials, or none.")
    coded = [standardize_trials(x)[0] for x in series]
    lengths = [[len(t) for t in c] for c in coded]
    if any(ls != lengths[0] for ls in lengths):
        raise ValueError("`source` and `target` (and `conditions`) must have the same length.")
    m = max(k, 1)
    t0 = max(k, l + lag - 1, m if conditions else 0)
    source_rows, target_rows, present, condition_rows = [], [], [], [[] for _ in conditions]
    for trial in range(len(coded[0])):
        N = lengths[0][trial]
        if t0 >= N:
            continue
        x, y = coded[0][trial], coded[1][trial]
        source_rows.append(_window_codes(x, l, t0 - lag - l + 1, N - lag - l + 1))
        target_rows.append(_window_codes(y, k, t0 - k, N - k))
        present.append(y[t0:N])
        for rows, z in zip(condition_rows, coded[2:], strict=True):
            rows.append(_window_codes(z[trial], m, t0 - m, N - m))
    if not present:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    source_past = _dense_rows(np.concatenate(source_rows))
    context = _dense_rows(np.concatenate(target_rows))
    for rows in condition_rows:
        context = _pair(context, _dense_rows(np.concatenate(rows)))
    return source_past, _dense(np.concatenate(present)), context


_TE_PARAMETERS = """source, target : array_like or Trials
        Equal-length series; rows of 2D arrays are joint symbols. Pass
        :class:`~dit.inference.Trials` of equal-length pairs to pool trials.
    history_length : int or 'auto'
        The target history length :math:`k \\geq 0`. ``'auto'`` uses the
        target's Markov order from :func:`~dit.inference.select_markov_order`
        (up to `max_history`).
    source_history : int, None
        The source history length :math:`l \\geq 1`; defaults to
        ``max(k, 1)``.
    lag : int
        The source past ends `lag` steps before the target's present.
    conditions : sequence of array_like, optional
        Further series whose pasts (of length ``max(k, 1)``) are conditioned on,
        giving the conditional transfer entropy."""


def transfer_entropy(
    source,
    target,
    history_length=1,
    estimator="plugin",
    source_history=None,
    lag=1,
    conditions=(),
    max_history=None,
    prng=None,
):
    """
    Estimate the transfer entropy from `source` to `target` in bits,

    .. math::

        T_{X \\to Y} = I[Y_t : X_{t-\\tau-l+1:t-\\tau+1} \\mid Y_{t-k:t}, Z_{t-m:t}]

    with :math:`k` = `history_length`, :math:`l` = `source_history`,
    :math:`\\tau` = `lag` and optional conditioning series :math:`Z`
    :cite:`Schreiber2000`. With the defaults this is
    :math:`I[Y_t : X_{t-k:t} \\mid Y_{t-k:t}]`.

    Parameters
    ----------
    {params}
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    max_history : int, None
        The largest history considered for ``history_length='auto'``.
    prng : None, int, Generator, RandomState
        Source of randomness for ``history_length='auto'``.

    Returns
    -------
    te : float
    """
    k, l, lag = _resolve_te(target, history_length, source_history, lag, max_history, prng)
    codes = _te_codes(source, target, k, l, lag, conditions)
    _check_joint(*codes)
    return _cmi_codes(*codes, estimator)


def transfer_entropy_test(
    source,
    target,
    history_length=1,
    null="conditional",
    n_surrogates=1000,
    estimator="plugin",
    surrogate_order=None,
    block_length=None,
    prng=None,
    source_history=None,
    lag=1,
    conditions=(),
    max_history=None,
):
    """
    Test whether the transfer entropy from `source` to `target` is zero.

    Parameters
    ----------
    {params}
    null : {'conditional', 'whittle', 'shift', 'block'}
        How surrogates are built:

        * ``'conditional'`` — permute the source past within strata of the target
          past (and the conditions' pasts). This targets exactly
          :math:`T_{X \\to Y} = 0`.
        * ``'whittle'`` — replace the source by :func:`~dit.inference.whittle_surrogates`
          of order `surrogate_order`, preserving its own Markov structure
          :cite:`Pethel2014`.
        * ``'shift'`` — circularly shift the source relative to the target.
        * ``'block'`` — shuffle blocks of `block_length` of the source.

        The last three test the stronger null that the source is independent of
        the target, while keeping the source's own memory. Unlike an i.i.d.
        shuffle, they do not inflate false positives when the source is
        autocorrelated.
    n_surrogates : int
        The number of surrogates.
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    surrogate_order : int, None
        The order preserved by ``'whittle'``; defaults to `source_history`.
    block_length : int, None
        The block length for ``'block'``; defaults to
        ``max(2 * (l + lag), round(sqrt(N)))``.
    prng : None, int, Generator, RandomState
        Source of randomness.
    max_history : int, None
        The largest history considered for ``history_length='auto'``.

    Returns
    -------
    result : SurrogateTest
    """
    rng = as_generator(prng)
    k, l, lag = _resolve_te(target, history_length, source_history, lag, max_history, rng)
    source_past, present, context = _te_codes(source, target, k, l, lag, conditions)
    _check_joint(source_past, present, context)
    value = _cmi_codes(source_past, present, context, estimator)
    n_samples = len(present)

    if null == "conditional":
        null_values = [
            _cmi_codes(_within_strata(source_past, context, rng), present, context, estimator)
            for _ in range(n_surrogates)
        ]
        return _result(value, np.array(null_values), n_samples)

    if not is_trials(source):
        source = np.asarray(source)
    if null == "whittle":
        order = l if surrogate_order is None else surrogate_order
        surrogates = whittle_surrogates(source, order, n=n_surrogates, prng=rng)
    elif null == "shift":
        surrogates = shift_surrogates(source, n=n_surrogates, prng=rng)
    elif null == "block":
        if block_length is None:
            N = min(len(t) for t in source) if is_trials(source) else len(source)
            block_length = max(2 * (l + lag), round(np.sqrt(N)))
        surrogates = block_surrogates(source, block_length, n=n_surrogates, prng=rng)
    else:
        raise ValueError(f"Unknown null {null!r}.")
    null_values = [_cmi_codes(*_te_codes(s, target, k, l, lag, conditions), estimator) for s in surrogates]
    return _result(value, np.array(null_values), n_samples)


def stationary_bootstrap(data, n=1, mean_block_length=None, prng=None):
    """
    Resample a series with the stationary bootstrap :cite:`Politis1994`.

    Each resample concatenates blocks that start at uniformly random positions
    and have geometrically distributed lengths, wrapping around the end of the
    series. Rows of multivariate `data` are resampled together. For
    :class:`~dit.inference.Trials`, whole trials are resampled with replacement
    instead (a cluster bootstrap), so no resample joins two trials.

    Parameters
    ----------
    data : array_like or Trials
        The series (along the first axis), or independent trials.
    n : int
        The number of resamples.
    mean_block_length : float, None
        The mean block length; defaults to ``N ** (1 / 3)``.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    resamples : np.ndarray or list of Trials
        Shape ``(n,) + np.shape(data)``, or a list of `n` :class:`Trials`.
    """
    rng = as_generator(prng)
    if is_trials(data):
        picks = rng.integers(0, len(data), size=(n, len(data)))
        return [Trials(data[j] for j in row) for row in picks]
    data = np.asarray(data)
    N = len(data)
    if mean_block_length is None:
        mean_block_length = N ** (1 / 3)
    new_block = rng.random((n, N)) < 1 / mean_block_length
    new_block[:, 0] = True
    starts = rng.integers(0, N, size=(n, N))
    positions = np.arange(N)
    block_start = np.maximum.accumulate(np.where(new_block, positions, 0), axis=1)
    index = (np.take_along_axis(starts, block_start, axis=1) + positions - block_start) % N
    return data[index]


def bootstrap_ci(data, statistic, n_boot=1000, confidence=0.95, mean_block_length=None, method="percentile", prng=None):
    """
    A confidence interval for `statistic` under the stationary bootstrap.

    Parameters
    ----------
    data : array_like
        The series (along the first axis). For statistics of several series,
        stack them as columns so they are resampled jointly.
    statistic : callable
        Maps a series shaped like `data` to a float, e.g.
        ``lambda d: transfer_entropy(d[:, 0], d[:, 1])``.
    n_boot : int
        The number of resamples.
    confidence : float
        The coverage of the interval.
    mean_block_length : float, None
        See :func:`stationary_bootstrap`.
    method : {'basic', 'percentile'}
        ``'percentile'`` returns quantiles of the resampled statistic.
        ``'basic'`` reflects them about the observed value,
        :math:`(2\\hat\\theta - q_{hi}, 2\\hat\\theta - q_{lo})`, which corrects
        for the shift between resampled and observed estimates.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    low, high : float
        The interval endpoints.

    Notes
    -----
    Every block junction in a resampled series creates length-``L`` words that
    straddle two unrelated positions, so statistics built from lagged windows
    (block entropies, transfer entropy) are biased toward independence unless
    `mean_block_length` is much larger than ``L``. For transfer entropy prefer
    :func:`transfer_entropy_ci`, which resamples the aligned windows instead.
    """
    if not is_trials(data):
        data = np.asarray(data)
    samples = [statistic(r) for r in stationary_bootstrap(data, n_boot, mean_block_length, prng)]
    return _interval(samples, statistic(data) if method == "basic" else None, confidence, method)


def _interval(samples, value, confidence, method):
    if method not in ("basic", "percentile"):
        raise ValueError(f"Unknown method {method!r}.")
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    if method == "basic":
        low, high = 2 * value - high, 2 * value - low
    return float(low), float(high)


def transfer_entropy_ci(
    source,
    target,
    history_length=1,
    n_boot=1000,
    confidence=0.95,
    mean_block_length=None,
    estimator="plugin",
    method="percentile",
    prng=None,
    source_history=None,
    lag=1,
    conditions=(),
    max_history=None,
):
    """
    A stationary-bootstrap confidence interval for the transfer entropy.

    The aligned samples (source past, target present, target and condition
    pasts) are resampled in blocks :cite:`Politis1994`, so block junctions never
    create windows that were not observed.

    Parameters
    ----------
    {params}
    n_boot : int
        The number of resamples.
    confidence : float
        The coverage of the interval.
    mean_block_length : float, None
        See :func:`stationary_bootstrap`.
    estimator : str
        See :func:`~dit.inference.entropy_from_counts`.
    method : {'percentile', 'basic'}
        See :func:`bootstrap_ci`.
    prng : None, int, Generator, RandomState
        Source of randomness.
    max_history : int, None
        The largest history considered for ``history_length='auto'``.

    Returns
    -------
    low, high : float
        The interval endpoints.
    """
    rng = as_generator(prng)
    k, l, lag = _resolve_te(target, history_length, source_history, lag, max_history, rng)
    source_past, present, context = _te_codes(source, target, k, l, lag, conditions)
    _check_joint(source_past, present, context)
    value = _cmi_codes(source_past, present, context, estimator)
    resamples = stationary_bootstrap(np.arange(len(present)), n_boot, mean_block_length, rng)
    samples = [_cmi_codes(source_past[i], present[i], context[i], estimator) for i in resamples]
    return _interval(samples, value, confidence, method)


def benjamini_hochberg(pvalues, alpha=0.05, dependent=False):
    """
    Control the false discovery rate over a fixed family of tests.

    Parameters
    ----------
    pvalues : array_like
        One p-value per test, e.g. every pairwise :func:`transfer_entropy_test`
        in a network.
    alpha : float
        The target false discovery rate.
    dependent : bool
        If False, the Benjamini–Hochberg step-up procedure, valid for
        independent or positively dependent tests :cite:`Benjamini1995`. If
        True, the Benjamini–Yekutieli correction, valid under any dependence
        :cite:`Benjamini2001`.

    Returns
    -------
    reject : np.ndarray of bool
        Which hypotheses are rejected.
    adjusted : np.ndarray
        Adjusted p-values (q-values); ``reject == (adjusted <= alpha)``.

    Notes
    -----
    The family must be fixed before looking at the results. Procedures that
    choose later tests from earlier outcomes, like CSSR's state splitting, are
    not covered.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    scale = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    ranked = p[order] * m * scale / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted <= alpha, adjusted


for _function in (transfer_entropy, transfer_entropy_test, transfer_entropy_ci):
    _function.__doc__ = _function.__doc__.replace("{params}", _TE_PARAMETERS)
del _function
