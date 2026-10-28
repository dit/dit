"""
Various methods for estimating information quantities from samples.
"""

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from ..utils.deprecation import renamed_kwarg
from ._advanced_estimators import _bub, _panzeri_treves, _pym, _quadratic_extrapolation, _unseen, _wu_yang
from ._symbols import check_sampling, standardize_trials, word_counts
from .counts import get_counts

__all__ = (
    "ENTROPY_ESTIMATORS",
    "block_entropy",
    "conditional_entropy_rate",
    "entropy_0",
    "entropy_1",
    "entropy_2",
    "entropy_from_counts",
    "conditional_mutual_information",
    "jackknife_entropy",
    "dispersion_entropy",
    "lz76_complexity",
    "lz_entropy_rate",
    "multiscale_permutation_entropy",
    "permutation_entropy",
    "weighted_permutation_entropy",
)


def entropy_0(data, length=1):
    """
    Estimate the entropy of length `length` subsequences in `data`.

    Parameters
    ----------
    data : iterable
        An iterable of samples.
    length : int
        The length to group samples into.

    Returns
    -------
    h0 : float
        An estimate of the entropy.

    Notes
    -----
    This returns the naive estimate of the entropy.
    """
    counts = get_counts(data, length)
    probs = counts / counts.sum()
    h0 = -np.nansum(probs * np.log2(probs))
    return h0


def entropy_1(data, length=1):
    """
    Estimate the entropy of length `length` subsequences in `data`.

    Parameters
    ----------
    data : iterable
        An iterable of samples.
    length : int
        The length to group samples into.

    Returns
    -------
    h1 : float
        An estimate of the entropy.

    Notes
    -----
    This is the digamma correction
    :math:`\\hat{H} = \\psi(N) - \\sum_i \\frac{n_i}{N} \\psi(n_i)`, the leading
    term of Grassberger's estimator :cite:`Grassberger1988` (not the
    Miller–Madow correction; see :func:`entropy_from_counts`).

    If M is the alphabet size and N is the number of samples, then the bias of this estimator is:
        B ~ M/N
    """
    counts = get_counts(data, length)
    total = counts.sum()
    digamma_N = digamma(total)

    h1 = np.log2(np.e) * (counts / total * (digamma_N - digamma(counts))).sum()

    return h1


def entropy_2(data, length=1):
    """
    Estimate the entropy of length `length` subsequences in `data`.

    Parameters
    ----------
    data : iterable
        An iterable of samples.
    length : int
        The length to group samples into.

    Returns
    -------
    h2 : float
        An estimate of the entropy.

    Notes
    -----
    This is Grassberger's (2003) estimator :cite:`Grassberger2003`,
    :math:`\\hat{H} = \\psi(N) - \\sum_i \\frac{n_i}{N} G(n_i)` with
    :math:`G(n) = \\psi(n) + \\tfrac{1}{2}(-1)^n [\\psi(\\tfrac{n+1}{2}) - \\psi(\\tfrac{n}{2})]`,
    using :math:`\\psi(N)` in place of Grassberger's :math:`\\ln N`. It is not
    the Nemenman–Shafee–Bialek (NSB) estimator.

    If M is the alphabet size and N is the number of samples, then the bias of this estimator is:
        B ~ (M+1)/(2N)
    """
    counts = get_counts(data, length)
    total = counts.sum()
    digamma_N = digamma(total)
    log2 = np.log(2)
    jss = [np.arange(1, count) for count in counts]

    alt_terms = np.array([(((-1) ** js) / js).sum() for js in jss])

    h2 = np.log2(np.e) * (counts / total * (digamma_N - digamma(counts) + log2 + alt_terms)).sum()

    return h2


def _plugin(counts, N, K=None):
    p = counts / N
    return -np.sum(p * np.log(p))


def _miller_madow(counts, N, K=None):
    return _plugin(counts, N) + (len(counts) - 1) / (2 * N)


def _digamma(counts, N, K=None):
    return digamma(N) - np.sum(counts * digamma(counts)) / N


def _grassberger(counts, N, K=None):
    G = digamma(counts) + 0.5 * (-1) ** counts * (digamma((counts + 1) / 2) - digamma(counts / 2))
    return np.log(N) - np.sum(counts * G) / N


def _chao_shen(counts, N, K=None):
    singletons = np.sum(counts == 1)
    if singletons == N:
        singletons = N - 1
    coverage = 1 - singletons / N
    pa = coverage * counts / N
    return -np.sum(pa * np.log(pa) / (1 - (1 - pa) ** N))


def _nsb(counts, N, K=None):
    """
    The Nemenman–Shafee–Bialek estimate (nats): the posterior mean entropy under a
    mixture of symmetric Dirichlet(beta) priors weighted so that the prior on the
    entropy itself is uniform on ``[0, ln K]``.
    """
    K = len(counts) if K is None else K
    if len(counts) > K:
        raise ValueError("`alphabet_size` is smaller than the number of observed outcomes.")
    if K == 1:
        return 0.0
    unseen = K - len(counts)
    log_beta = np.linspace(-12.0, 8.0, 4001)
    beta = np.exp(log_beta)
    Kb = K * beta
    # log evidence p(counts | beta), up to a constant
    log_evidence = (
        gammaln(Kb)
        - gammaln(N + Kb)
        + np.sum(gammaln(counts[:, None] + beta[None, :]), axis=0)
        - len(counts) * gammaln(beta)
    )
    # d E[H | beta] / d beta, so that the implied prior on H is uniform
    log_prior = np.log(K * polygamma(1, Kb + 1) - polygamma(1, beta + 1))
    log_w = log_evidence + log_prior + log_beta  # d beta = beta d log beta
    w = np.exp(log_w - log_w.max())
    posterior_mean = digamma(N + Kb + 1) - (
        np.sum((counts[:, None] + beta[None, :]) * digamma(counts[:, None] + beta[None, :] + 1), axis=0)
        + unseen * beta * digamma(beta + 1)
    ) / (N + Kb)
    return float(np.sum(w * posterior_mean) / np.sum(w))


def _alphabet(counts, K):
    K = len(counts) if K is None else K
    if len(counts) > K:
        raise ValueError("`alphabet_size` is smaller than the number of observed outcomes.")
    return K


def _plugin_with_unseen(p_seen, p_unseen, unseen):
    """
    The entropy (nats) of `p_seen` plus `unseen` outcomes of probability `p_unseen` each.
    """
    h = -np.sum(p_seen * np.log(p_seen))
    if unseen and p_unseen > 0:
        h -= unseen * p_unseen * np.log(p_unseen)
    return h


def _zhang(counts, N, K=None):
    """
    Zhang's estimator, computed as in Lozano et al.: for each distinct count
    :math:`n`, :math:`\\sum_{v=1}^{N-n} \\frac{1}{v} \\prod_{j=0}^{v-1} \\frac{N-n-j}{N-1-j}`.
    """
    N = int(N)
    h = 0.0
    for n, mult in zip(*np.unique(counts.astype(np.int64), return_counts=True), strict=True):
        j = np.arange(N - n)
        term = np.sum(np.cumprod((N - n - j) / (N - 1 - j)) / (j + 1))
        h += mult * n / N * term
    return h


def _harmonic(n):
    """
    The harmonic numbers :math:`H_n = \\sum_{k=1}^{n} 1/k`, elementwise.
    """
    return digamma(np.asarray(n, dtype=float) + 1) + np.euler_gamma


def _chao_wang_jost(counts, N, K=None):
    """
    The Chao–Wang–Jost estimator, from the discovery rate of new outcomes.
    """
    N = int(N)
    f1, f2 = np.sum(counts == 1), np.sum(counts == 2)
    if f2 > 0:
        A = 2 * f2 / ((N - 1) * f1 + 2 * f2)
    elif f1 > 0:
        A = 2 / ((N - 1) * (f1 - 1) + 2)
    else:
        A = 1.0
    seen = counts[counts < N]
    h = np.sum(seen / N * (_harmonic(N - 1) - _harmonic(seen - 1)))
    if f1 == 0 or A >= 1:
        return h
    return h + f1 / N * _cwj_tail(A, N)


def _cwj_tail(A, N, series=None):
    """
    :math:`(1 - A)^{1 - N} (-\\ln A - \\sum_{r=1}^{N-1} (1 - A)^r / r)`, which also
    equals :math:`\\sum_{m \\geq 1} (1 - A)^m / (N + m - 1)`. The closed form
    cancels catastrophically (and overflows) unless :math:`A N` is small, so it is
    used only when :math:`A` is too small for the series to converge quickly.
    """
    log_q = np.log1p(-A)
    if series is None:
        series = -log_q * 1e6 >= 40
    if series:
        m = np.arange(1, int(np.ceil(-40 / log_q)) + 1)
        return np.sum(np.exp(m * log_q) / (N + m - 1))
    r = np.arange(1, N)
    return np.exp((1 - N) * log_q) * (-np.log(A) - np.sum(np.exp(r * log_q) / r))


def _shrink(counts, N, K=None):
    """
    The James–Stein shrinkage estimator: the plug-in entropy of the empirical
    distribution shrunk toward uniform with the Hausser–Strimmer intensity.
    """
    K = _alphabet(counts, K)
    u = counts / N
    unseen = K - len(counts)
    t = 1 / K
    spread = np.sum((t - u) ** 2) + unseen * t**2
    lam = 1.0 if N <= 1 or spread == 0 else float(np.clip((1 - np.sum(u**2)) / ((N - 1) * spread), 0, 1))
    return _plugin_with_unseen(lam * t + (1 - lam) * u, lam * t, unseen)


def _bayes(alpha):
    """
    The plug-in entropy of the posterior mean under a symmetric Dirichlet prior.
    `alpha` maps ``(N, K)`` to the concentration parameter.
    """

    def estimator(counts, N, K=None):
        K = _alphabet(counts, K)
        a = alpha(N, K)
        total = N + K * a
        return _plugin_with_unseen((counts + a) / total, a / total, K - len(counts))

    return estimator


def _ansb(counts, N, K=None):
    """
    The asymptotic NSB estimate, :math:`(\\gamma - \\ln 2) + 2 \\ln N - \\psi(\\Delta)`
    with :math:`\\Delta` the number of coincidences.
    """
    coincidences = N - len(counts)
    if coincidences <= 0:
        raise ValueError("The ANSB estimator needs at least one repeated outcome.")
    return np.euler_gamma - np.log(2) + 2 * np.log(N) - digamma(coincidences)


def _bonachela(counts, N, K=None):
    """
    Bonachela et al.'s balanced estimator over the observed outcomes.
    """
    N = int(N)
    # sum_{j = n + 2}^{N + 2} 1/j = H_{N + 2} - H_{n + 1}
    return np.sum((counts + 1) * (_harmonic(N + 2) - _harmonic(counts + 1))) / (N + 2)


def _good_turing(counts, N, K=None):
    """
    The plug-in entropy of the simple Good–Turing probabilities of Gale and
    Sampson. The unseen mass :math:`N_1 / N` is shared equally among the
    `K - len(counts)` unseen outcomes, or dropped (and the rest renormalized)
    when there are none.
    """
    K = _alphabet(counts, K)
    r, n_r = np.unique(counts.astype(np.int64), return_counts=True)
    if len(r) < 2:
        return _plugin(counts, N)
    q = np.concatenate([[0], r[:-1]])
    t = np.concatenate([r[1:], [2 * r[-1] - q[-1]]])
    z = n_r / (0.5 * (t - q))
    slope, intercept = np.polyfit(np.log(r), np.log(z), 1)
    observed = dict(zip(r.tolist(), n_r.tolist(), strict=True))
    r_star = np.empty(len(r))
    use_turing = True
    for i, (ri, ni) in enumerate(zip(r, n_r, strict=True)):
        lgt = (ri + 1) * (1 + 1 / ri) ** slope
        nxt = observed.get(int(ri) + 1)
        if use_turing and nxt is not None:
            turing = (ri + 1) * nxt / ni
            if abs(turing - lgt) > 1.96 * np.sqrt((ri + 1) ** 2 * nxt / ni**2 * (1 + nxt / ni)):
                r_star[i] = turing
                continue
        use_turing = False
        r_star[i] = lgt
    unseen = K - len(counts)
    p0 = observed.get(1, 0) / N if unseen else 0.0
    p_r = (1 - p0) * r_star / np.sum(n_r * r_star)
    p_seen = p_r[np.searchsorted(r, counts.astype(np.int64))]
    return _plugin_with_unseen(p_seen, p0 / unseen if unseen else 0.0, unseen)


def _jackknife(estimator):
    """
    The delete-one jackknife bias correction of `estimator`.
    """

    def corrected(counts, N, K=None):
        return _jackknife_parts(estimator, counts, N, K)[0]

    return corrected


def _jackknife_parts(estimator, counts, N, K):
    """
    The jackknifed estimate (nats) and the jackknife variance (nats squared).
    """
    if N < 2:
        return estimator(counts, N, K), 0.0
    loo = np.empty(len(counts))
    for i in range(len(counts)):
        reduced = counts.copy()
        reduced[i] -= 1
        loo[i] = estimator(reduced[reduced > 0], N - 1, K)
    mean = np.sum(counts * loo) / N
    estimate = N * estimator(counts, N, K) - (N - 1) * mean
    variance = (N - 1) / N * np.sum(counts * (loo - mean) ** 2)
    return estimate, variance


ENTROPY_ESTIMATORS = {
    "plugin": _plugin,
    "miller_madow": _miller_madow,
    "digamma": _digamma,
    "grassberger": _grassberger,
    "chao_shen": _chao_shen,
    "nsb": _nsb,
    "zhang": _zhang,
    "chao_wang_jost": _chao_wang_jost,
    "shrink": _shrink,
    "bayes_jeffreys": _bayes(lambda N, K: 0.5),
    "bayes_laplace": _bayes(lambda N, K: 1.0),
    "bayes_perks": _bayes(lambda N, K: 1 / K),
    "bayes_minimax": _bayes(lambda N, K: np.sqrt(N) / K),
    "ansb": _ansb,
    "bonachela": _bonachela,
    "good_turing": _good_turing,
    "jackknife": _jackknife(_plugin),
    "panzeri_treves": _panzeri_treves,
    "quadratic_extrapolation": _quadratic_extrapolation,
    "wu_yang": _wu_yang,
    "bub": _bub,
    "unseen": _unseen,
    "pym": _pym,
}


def entropy_from_counts(counts, estimator="plugin", alphabet_size=None):
    """
    Estimate an entropy, in bits, from a vector of outcome counts.

    Parameters
    ----------
    counts : array_like
        The number of times each outcome was observed. Zeros are ignored.
    estimator : str
        One of:

        * ``'plugin'`` — the maximum-likelihood (naive) estimate.
        * ``'miller_madow'`` — plug-in plus :math:`(K - 1) / 2N` nats for
          :math:`K` observed outcomes :cite:`Miller1955`.
        * ``'digamma'`` — :math:`\\psi(N) - \\sum_i \\frac{n_i}{N} \\psi(n_i)`
          :cite:`Grassberger1988` (as :func:`entropy_1`).
        * ``'grassberger'`` — Grassberger's (2003) estimator
          :cite:`Grassberger2003`.
        * ``'chao_shen'`` — coverage-adjusted Horvitz–Thompson estimator,
          which accounts for unseen outcomes :cite:`Chao2003`.
        * ``'nsb'`` — the Nemenman–Shafee–Bialek Bayesian estimator, which
          averages Dirichlet priors so that the prior on the entropy is
          nearly uniform :cite:`Nemenman2002`. Needs `alphabet_size`.
        * ``'zhang'`` — Zhang's estimator, whose bias decays exponentially in
          :math:`N` :cite:`Zhang2012`, computed as in :cite:`Lozano2017`.
        * ``'chao_wang_jost'`` — the estimator of Chao, Wang & Jost, built from
          the rate at which new outcomes are discovered :cite:`Chao2013`.
        * ``'shrink'`` — the plug-in entropy of the James–Stein shrinkage of the
          empirical distribution toward uniform :cite:`Hausser2009`. Needs
          `alphabet_size`.
        * ``'bayes_jeffreys'``, ``'bayes_laplace'``, ``'bayes_perks'``,
          ``'bayes_minimax'`` — the plug-in entropy of the posterior mean under
          a symmetric Dirichlet prior with concentration :math:`1/2`
          (Krichevsky–Trofimov :cite:`Krichevsky1981`), :math:`1`,
          :math:`1/K`, and :math:`\\sqrt{N}/K` :cite:`Hausser2009`. Need
          `alphabet_size`.
        * ``'ansb'`` — the asymptotic NSB estimate
          :math:`(\\gamma - \\ln 2) + 2 \\ln N - \\psi(\\Delta)`, with
          :math:`\\Delta` the number of coincidences (samples minus distinct
          outcomes), for :math:`N \\ll K` :cite:`Nemenman2004`. It raises if
          there are no coincidences and is badly biased when :math:`N` is
          comparable to :math:`K`.
        * ``'bonachela'`` — Bonachela, Hinrichsen & Muñoz's estimator, which
          balances bias and variance on short series rather than minimizing
          bias :cite:`Bonachela2008`. It sums over the observed outcomes.
        * ``'good_turing'`` — the plug-in entropy of the simple Good–Turing
          probabilities :cite:`Gale1995`. The unseen mass :math:`N_1/N` is
          shared among the unseen outcomes implied by `alphabet_size`.
        * ``'jackknife'`` — the delete-one jackknife of the plug-in estimate
          :cite:`Efron1981`; see :func:`jackknife_entropy` to jackknife any
          estimator.
        * ``'panzeri_treves'`` — plug-in plus :math:`(\\tilde{R} - 1)/2N`, with
          :math:`\\tilde{R}` a Bayesian count of the relevant outcomes
          :cite:`Panzeri1996`.
        * ``'quadratic_extrapolation'`` — fits
          :math:`H_\\infty + b/n + c/n^2` to the plug-in entropy at
          :math:`n = N, N/2, N/4` :cite:`Strong1998`, using the exact expected
          entropy of random subsamples rather than a particular split.
        * ``'wu_yang'`` — the minimax-rate estimator built from the best
          polynomial approximation of :math:`-p \\ln p` :cite:`Wu2016`. Its
          degree, :math:`\\lfloor 1.6 \\ln K \\rfloor`, is capped at 22.
        * ``'bub'`` — Paninski's best-upper-bound estimator, linear in the
          count histogram, minimizing a bound on the worst-case error
          :cite:`Paninski2003`. It is meant for :math:`N/K \\lesssim 100`.
        * ``'unseen'`` — the entropy of the probability histogram recovered by
          linear programming from the fingerprint :cite:`Valiant2017`.
        * ``'pym'`` — the posterior mean under a mixture of Pitman–Yor
          process priors, suited to heavy-tailed distributions on unbounded
          alphabets :cite:`Archer2014`. It raises without repeated outcomes.
    alphabet_size : int, None
        The number of possible outcomes :math:`K`, used by ``'nsb'``,
        ``'shrink'``, the ``'bayes_*'`` estimators, ``'good_turing'``,
        ``'panzeri_treves'``, ``'wu_yang'`` and ``'bub'``. Defaults to ``len(counts)``, so pass a count vector
        that includes the zero counts or set this explicitly.

    Returns
    -------
    h : float
        The estimated entropy in bits.

    Warns
    -----
    UndersamplingWarning
        If more than a fifth as many outcomes were observed as samples.
    """
    counts = np.asarray(counts, dtype=float)
    check_sampling(int(np.sum(counts > 0)), int(counts.sum()))
    return _entropy(counts, estimator, len(counts) if alphabet_size is None else alphabet_size)


def _entropy(counts, estimator, alphabet_size=None):
    """
    :func:`entropy_from_counts` without the sampling check, for internal loops.
    """
    try:
        fn = ENTROPY_ESTIMATORS[estimator]
    except KeyError:
        raise ValueError(f"Unknown estimator {estimator!r}; choose from {sorted(ENTROPY_ESTIMATORS)}.") from None
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    N = counts.sum()
    if N == 0:
        return 0.0
    return float(fn(counts, N, alphabet_size) / np.log(2))


def jackknife_entropy(counts, estimator="plugin", alphabet_size=None):
    """
    The delete-one jackknife of an entropy estimator, with its variance.

    With :math:`\\hat{H}_{-i}` the estimate after removing one sample of outcome
    :math:`i`, the jackknifed estimate is
    :math:`N \\hat{H} - \\frac{N-1}{N} \\sum_i n_i \\hat{H}_{-i}`, which removes the
    :math:`1/N` term of the bias, and the jackknife variance is
    :math:`\\frac{N-1}{N} \\sum_i n_i (\\hat{H}_{-i} - \\bar{H})^2` :cite:`Efron1981`.

    Parameters
    ----------
    counts : array_like
        The number of times each outcome was observed. Zeros are ignored.
    estimator : str
        The estimator to jackknife; see :func:`entropy_from_counts`.
    alphabet_size : int, None
        See :func:`entropy_from_counts`.

    Returns
    -------
    h : float
        The jackknifed entropy estimate, in bits.
    variance : float
        The jackknife estimate of its variance, in bits squared.
    """
    try:
        fn = ENTROPY_ESTIMATORS[estimator]
    except KeyError:
        raise ValueError(f"Unknown estimator {estimator!r}; choose from {sorted(ENTROPY_ESTIMATORS)}.") from None
    counts = np.asarray(counts, dtype=float)
    K = len(counts) if alphabet_size is None else alphabet_size
    counts = counts[counts > 0]
    N = counts.sum()
    if N == 0:
        return 0.0, 0.0
    h, var = _jackknife_parts(fn, counts, N, K)
    return float(h / np.log(2)), float(var / np.log(2) ** 2)


def block_entropy(data, length, estimator="plugin"):
    """
    Estimate :math:`H[X_{0:L}]`, the entropy of overlapping length-`length` words.

    Parameters
    ----------
    data : iterable or Trials
        The sequence, or independent :class:`~dit.inference.Trials` (words never
        span two trials). Rows of a 2D array are joint symbols.
    length : int
        The word length :math:`L`.
    estimator : str
        See :func:`entropy_from_counts`.

    Returns
    -------
    h : float
        The estimated block entropy in bits.
    """
    trials, alphabet = standardize_trials(data)
    counts = word_counts(trials, length, len(alphabet))
    check_sampling(len(counts), int(counts.sum()), len(alphabet), length)
    return _entropy(counts, estimator, len(alphabet) ** length)


def conditional_entropy_rate(data, L, estimator="plugin"):
    """
    Estimate :math:`h_L = H[X_L \\mid X_{0:L}] = H[X_{0:L+1}] - H[X_{0:L}]`.

    As `L` grows, :math:`h_L` decreases to the entropy rate; for a Markov chain of
    order :math:`R` it equals the entropy rate for all :math:`L \\geq R`
    :cite:`Cover2006`. Use :func:`~dit.inference.select_markov_order` to choose
    `L`. Bias-corrected `estimator` choices matter most when the number of
    observed words is not small compared to the sample size.

    Parameters
    ----------
    data : iterable or Trials
        The sequence, or independent :class:`~dit.inference.Trials` (words never
        span two trials). Rows of a 2D array are joint symbols.
    L : int
        The history length.
    estimator : str
        See :func:`entropy_from_counts`.

    Returns
    -------
    h : float
        The estimated conditional entropy in bits.
    """
    trials, alphabet = standardize_trials(data)
    M = len(alphabet)
    joint = word_counts(trials, L + 1, M)
    check_sampling(len(joint), int(joint.sum()), M, L + 1)
    upper = _entropy(joint, estimator, M ** (L + 1))
    lower = _entropy(word_counts(trials, L, M), estimator, M**L) if L > 0 else 0.0
    return upper - lower


def lz_entropy_rate(data):
    """
    Estimate the entropy rate in bits per symbol from match lengths.

    For each position :math:`i`, :math:`\\Lambda_i` is the length of the shortest
    string starting at :math:`i` that does not appear starting at any earlier
    position. The increasing-window estimator of Kontoyiannis et al.
    :cite:`Kontoyiannis1998` is

    .. math::

        \\hat{h} = \\left[\\frac{1}{n} \\sum_{i} \\frac{\\Lambda_i}{\\log_2 (i + 1)}\\right]^{-1}.

    It is consistent for stationary ergodic processes and needs no history
    length. That makes it a check on :func:`conditional_entropy_rate`, whose
    answer depends on the chosen :math:`L`. Like the Lempel–Ziv compression
    rate :cite:`Cover2006`, it converges slowly: :math:`\\Lambda_i` exceeds
    :math:`\\log_2 i / h` by a roughly constant amount, so the estimate usually
    approaches the entropy rate from below (e.g. about 0.9 bits for a fair coin
    at :math:`n = 10^4`). Treat it as a cross-check rather than a precise
    estimate.

    Parameters
    ----------
    data : iterable or Trials
        The sequence. Rows of a 2D array are joint symbols. For
        :class:`~dit.inference.Trials`, the per-trial sums are pooled; matches
        are sought only within each trial.

    Returns
    -------
    h : float
        The estimated entropy rate in bits per symbol.

    Notes
    -----
    Positions whose match runs to the end of the data are excluded, since their
    :math:`\\Lambda_i` is censored. The cost is roughly
    :math:`O(n^2 \\log n)` character comparisons, done by ``str.find``.
    """
    trials, _ = standardize_trials(data)
    total = 0.0
    count = 0
    for codes in trials:
        text = "".join(map(chr, codes.tolist()))
        n = len(text)
        length = 1
        for i in range(1, n):
            length = max(1, length - 1)
            while i + length <= n and text.find(text[i : i + length], 0, i + length - 1) != -1:
                length += 1
            total += min(length, n - i + 1) / np.log2(i + 1)
            count += 1
    if count == 0 or total == 0:
        raise ValueError("`data` is too short to estimate an entropy rate.")
    return float(count / total)


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


def _cmi_codes(x, y, z, estimator, sizes=None):
    """
    :math:`I[x : y \\mid z]` in bits from dense integer codes. `sizes` gives the
    alphabet sizes of `x`, `y` and `z`; joint alphabets are their products.
    """
    Kx, Ky, Kz = sizes if sizes is not None else (x.max() + 1, y.max() + 1, z.max() + 1)

    def H(codes, K):
        return _entropy(np.bincount(codes), estimator, int(K))

    xz, yz = _pair(x, z), _pair(y, z)
    xyz = _pair(xz, y)
    return H(xz, Kx * Kz) + H(yz, Ky * Kz) - H(xyz, Kx * Ky * Kz) - H(z, Kz)


def _check_joint(*codes):
    """
    Warn if the joint outcomes of aligned code arrays are undersampled.
    """
    joint = codes[0]
    for c in codes[1:]:
        joint = _pair(joint, c)
    check_sampling(len(np.unique(joint)), len(joint), stacklevel=3)


def conditional_mutual_information(x, y, z=None, estimator="plugin", alphabet_sizes=None):
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
    alphabet_sizes : tuple of int, None
        The numbers of possible outcomes of `x`, `y` and `z`, used by estimators
        that need an alphabet size (such as ``'nsb'`` or ``'shrink'``); each
        joint term gets the product of its variables' sizes. Defaults to the
        number of distinct observed values of each variable.

    Returns
    -------
    cmi : float
    """
    x, y = _dense(x), _dense(y)
    z = np.zeros_like(x) if z is None else _dense(z)
    _check_joint(x, y, z)
    sizes = None
    if alphabet_sizes is not None:
        sizes = tuple(alphabet_sizes) + (1,) * (3 - len(alphabet_sizes))
        observed = (x.max() + 1, y.max() + 1, z.max() + 1)
        if any(s < o for s, o in zip(sizes, observed, strict=True)):
            raise ValueError("`alphabet_sizes` are smaller than the numbers of observed values.")
    return _cmi_codes(x, y, z, estimator, sizes)


@renamed_kwarg("prng", "rng")
def permutation_entropy(ts, order=3, delay=1, normalize=False, estimator="plugin", ties="first", rng=None):
    """
    The permutation entropy of a real-valued series :cite:`Bandt2002`.

    This is the entropy, in bits, of the distribution of
    :func:`~dit.inference.ordinal_patterns` of order :math:`m` and delay
    :math:`\\tau`. For piecewise-monotone maps, the permutation entropy per
    symbol converges to the Kolmogorov–Sinai entropy as :math:`m \\to \\infty`.

    Parameters
    ----------
    ts : array_like or Trials
        The series, or independent trials (patterns are pooled).
    order, delay : int
        The pattern order :math:`m` and delay :math:`\\tau`.
    normalize : bool
        Divide by :math:`\\log_2 m!`, giving a value in :math:`[0, 1]`.
    estimator : str
        See :func:`entropy_from_counts`; ``'nsb'`` uses :math:`m!` outcomes.
    ties : {'first', 'noise', 'distinct'}
        See :func:`~dit.inference.ordinal_patterns`.
    rng : None, int, Generator, RandomState
        Source of randomness for ``ties='noise'``.

    Returns
    -------
    h : float
    """
    from math import factorial

    from ._symbols import is_trials
    from .binning import ordinal_patterns

    patterns = ordinal_patterns(ts, order, delay, ties, rng)
    codes = np.concatenate(list(patterns)) if is_trials(patterns) else patterns
    K = factorial(order)
    counts = np.unique(codes, return_counts=True)[1]
    check_sampling(len(counts), int(counts.sum()), K, 1)
    h = _entropy(counts, estimator, K)
    return h / np.log2(K) if normalize else h


@renamed_kwarg("prng", "rng")
def weighted_permutation_entropy(ts, order=3, delay=1, normalize=False, ties="first", rng=None):
    """
    The weighted permutation entropy of a real-valued series :cite:`Fadlallah2013`.

    Each window's pattern is weighted by the variance of the window, so patterns
    of large fluctuations count more than patterns of noise-level wiggles. This
    restores some of the amplitude information that
    :func:`permutation_entropy` discards. Garland, James & Bradley use it as a
    model-free measure of predictability :cite:`Garland2014`.

    Parameters
    ----------
    ts : array_like or Trials
        The series, or independent trials (weights are pooled).
    order, delay : int
        The pattern order :math:`m` and delay :math:`\\tau`.
    normalize : bool
        Divide by :math:`\\log_2 m!`.
    ties : {'first', 'noise', 'distinct'}
        See :func:`~dit.inference.ordinal_patterns`.
    rng : None, int, Generator, RandomState
        Source of randomness for ``ties='noise'``.

    Returns
    -------
    h : float
    """
    from math import factorial

    from ._symbols import is_trials
    from .binning import _windows, ordinal_patterns

    series = list(ts) if is_trials(ts) else [ts]
    codes, weights = [], []
    for s in series:
        codes.append(ordinal_patterns(s, order, delay, ties, rng))
        weights.append(np.var(_windows(s, order, delay), axis=1))
    codes, weights = np.concatenate(codes), np.concatenate(weights)
    total = weights.sum()
    if total == 0:
        return 0.0
    p = np.bincount(np.unique(codes, return_inverse=True)[1].ravel(), weights=weights) / total
    p = p[p > 0]
    h = float(-np.sum(p * np.log2(p)))
    return h / np.log2(factorial(order)) if normalize else h


@renamed_kwarg("prng", "rng")
def multiscale_permutation_entropy(
    ts, scale, order=3, delay=1, normalize=False, estimator="plugin", ties="first", rng=None
):
    """
    The permutation entropy of a series coarse-grained at scales 1 through
    `scale` :cite:`Aziz2005`.

    At scale :math:`s` the series is replaced by the means of consecutive,
    non-overlapping blocks of :math:`s` values :cite:`Costa2002`, and
    :func:`permutation_entropy` is applied to the result.

    Parameters
    ----------
    ts : array_like
        The real-valued series.
    scale : int
        The largest scale.
    order, delay, normalize, estimator, ties, rng
        See :func:`permutation_entropy`.

    Returns
    -------
    h : np.ndarray
        The permutation entropy at each scale, in bits; entry ``s - 1`` is scale ``s``.
    """
    ts = np.asarray(ts, dtype=float)
    out = np.empty(scale)
    for s in range(1, scale + 1):
        blocks = len(ts) // s
        coarse = ts[: blocks * s].reshape(blocks, s).mean(axis=1)
        out[s - 1] = permutation_entropy(coarse, order, delay, normalize, estimator, ties, rng)
    return out


def dispersion_entropy(ts, classes=3, order=3, delay=1, mapping="ncdf", normalize=False, estimator="plugin"):
    """
    The dispersion entropy of a real-valued series :cite:`Rostaghi2016`.

    Each value is mapped to one of `classes` levels, through the normal CDF with
    the series' mean and standard deviation (``mapping='ncdf'``) or by min-max
    scaling (``'linear'``): a scaled value :math:`y \\in [0, 1]` gets class
    :math:`\\min(\\lfloor c y \\rfloor + 1, c)`. The dispersion entropy is the
    entropy of the length-`order` patterns of classes, taken `delay` apart.
    Unlike :func:`permutation_entropy` it keeps amplitude information and has
    no ties to break.

    Parameters
    ----------
    ts : array_like
        The real-valued series.
    classes : int
        The number of classes :math:`c`.
    order, delay : int
        The pattern length :math:`m` and the delay between its entries.
    mapping : {'ncdf', 'linear'}
        How values are scaled to :math:`[0, 1]` before classing.
    normalize : bool
        Divide by :math:`m \\log_2 c`, giving a value in :math:`[0, 1]`.
    estimator : str
        See :func:`entropy_from_counts`; alphabet-dependent estimators use
        :math:`c^m` patterns.

    Returns
    -------
    h : float
        The dispersion entropy, in bits. It is 0 for a constant series.
    """
    from scipy.special import ndtr

    ts = np.asarray(ts, dtype=float)
    if mapping == "ncdf":
        scale = np.std(ts)
        y = ndtr((ts - ts.mean()) / scale) if scale > 0 else None
    elif mapping == "linear":
        spread = ts.max() - ts.min()
        y = (ts - ts.min()) / spread if spread > 0 else None
    else:
        raise ValueError("`mapping` must be 'ncdf' or 'linear'.")
    if y is None:
        return 0.0
    z = np.minimum(np.floor(classes * y).astype(np.int64), classes - 1)
    n = len(z) - (order - 1) * delay
    if n < 1:
        raise ValueError("The series is too short for this `order` and `delay`.")
    patterns = np.stack([z[k * delay : k * delay + n] for k in range(order)], axis=1)
    counts = np.unique(patterns, axis=0, return_counts=True)[1]
    h = _entropy(counts, estimator, classes**order)
    return h / (order * np.log2(classes)) if normalize else h


def lz76_complexity(data, normalize=False):
    """
    The Lempel–Ziv (1976) complexity of a sequence :cite:`Lempel1976`.

    This is the number :math:`c(n)` of phrases in the exhaustive history of the
    sequence: scanning left to right, each phrase is the shortest string that
    does not occur earlier (overlaps allowed), computed with the algorithm of
    Kaspar & Schuster :cite:`Kaspar1987`. For example ``0001101001000101``
    parses as ``0 | 001 | 10 | 100 | 1000 | 101``, so :math:`c = 6`.

    Parameters
    ----------
    data : iterable
        The sequence; rows of a 2D array are joint symbols.
    normalize : bool
        Return :math:`c(n) \\log_k(n) / n` for an alphabet of :math:`k` observed
        symbols. For a stationary ergodic source this converges to the entropy
        rate divided by :math:`\\log_2 k`; random sequences score near 1. A
        constant sequence is normalized with :math:`k = 2`.

    Returns
    -------
    c : int or float
        The phrase count, or its normalization.
    """
    codes = _dense(data)
    n = len(codes)
    if n == 0:
        return 0
    c, start, i, k, k_max = 1, 1, 0, 1, 1
    while start + k <= n:
        if codes[i + k - 1] == codes[start + k - 1]:
            k += 1
            if start + k > n:
                c += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == start:
                c += 1
                start += k_max
                if start + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    if not normalize:
        return c
    return c * np.log(n) / np.log(max(len(np.unique(codes)), 2)) / n
