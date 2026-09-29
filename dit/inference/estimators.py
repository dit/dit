"""
Various methods for estimating information quantities from samples.
"""

import numpy as np
from scipy.special import digamma, gammaln, polygamma

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
    "lz_entropy_rate",
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


ENTROPY_ESTIMATORS = {
    "plugin": _plugin,
    "miller_madow": _miller_madow,
    "digamma": _digamma,
    "grassberger": _grassberger,
    "chao_shen": _chao_shen,
    "nsb": _nsb,
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
    alphabet_size : int, None
        The number of possible outcomes :math:`K`, used by ``'nsb'``. Defaults to
        ``len(counts)``, so pass a count vector that includes the zero counts or
        set this explicitly.

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


def _cmi_codes(x, y, z, estimator):
    """
    :math:`I[x : y \\mid z]` in bits from dense integer codes.
    """

    def H(codes):
        return _entropy(np.bincount(codes), estimator)

    xz, yz = _pair(x, z), _pair(y, z)
    xyz = _pair(xz, y)
    return H(xz) + H(yz) - H(xyz) - H(z)


def _check_joint(*codes):
    """
    Warn if the joint outcomes of aligned code arrays are undersampled.
    """
    joint = codes[0]
    for c in codes[1:]:
        joint = _pair(joint, c)
    check_sampling(len(np.unique(joint)), len(joint), stacklevel=3)


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
