"""
Entropy estimators for the undersampled regime that need numerical
optimization: Panzeri-Treves, quadratic extrapolation, Wu-Yang, BUB, Unseen,
and PYM. Each takes ``(counts, N, K)`` (positive counts, their total, and the
alphabet size) and returns nats.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import legendre
from numpy.polynomial import polynomial as P
from scipy.linalg import pinv
from scipy.optimize import linprog, minimize
from scipy.special import digamma, gammaln
from scipy.stats import hypergeom, poisson


def _plugin(counts, N):
    p = counts / N
    return -np.sum(p * np.log(p))


def _xlogx(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1)), 0.0)


def _panzeri_treves_support(counts, N, K):
    """
    The Bayesian estimate of the number of relevant outcomes of Panzeri & Treves.
    """
    p = counts / N
    R = len(p)
    if R >= K:
        return float(R)
    expected = R - np.sum((1 - p) ** N)
    delta_prev, delta = K, abs(R - expected)
    extra = 0.0
    while delta < delta_prev and R + extra < K:
        extra += 1.0
        gamma = extra * (1 - (N / (N + R)) ** (1 / N))
        p_seen = (1 - gamma) / (N + R) * (p * N + 1)
        expected = np.sum(1 - (1 - p_seen) ** N) + extra * (1 - (1 - gamma / extra) ** N)
        delta_prev, delta = delta, abs(R - expected)
    R = R + extra - 1.0
    if delta < delta_prev:
        R += 1.0
    return R


def _panzeri_treves(counts, N, K):
    """
    Plug-in plus :math:`(\\tilde{R} - 1) / 2N`, with :math:`\\tilde{R}` the
    Bayesian count of relevant outcomes.
    """
    K = len(counts) if K is None else K
    return _plugin(counts, N) + (_panzeri_treves_support(counts, N, K) - 1) / (2 * N)


def _expected_subsample_plugin(counts, N, n):
    """
    The expected plug-in entropy of a size-`n` subsample drawn without replacement.
    """
    values, mult = np.unique(counts.astype(np.int64), return_counts=True)
    x = np.arange(1, n + 1)
    pmf = hypergeom.pmf(x[None, :], int(N), values[:, None], n)
    return float(np.sum(mult * (pmf @ -_xlogx(x / n))))


def _quadratic_extrapolation(counts, N, K):
    """
    Fit :math:`H(n) = H_\\infty + b/n + c/n^2` to the expected plug-in entropy of
    subsamples of size :math:`N`, :math:`N/2` and :math:`N/4`, and return
    :math:`H_\\infty`.
    """
    N = int(N)
    if N < 4:
        return _plugin(counts, N)
    sizes = np.array([N, N // 2, N // 4], dtype=float)
    H = [_plugin(counts, N)] + [_expected_subsample_plugin(counts, N, int(n)) for n in sizes[1:]]
    design = np.stack([np.ones(3), 1 / sizes, 1 / sizes**2], axis=1)
    return float(np.linalg.solve(design, H)[0])


@lru_cache(maxsize=64)
def _best_approximation(degree):
    """
    Monomial coefficients of the best uniform polynomial approximation of
    :math:`-x \\ln x` on :math:`[0, 1]`, by the Remez exchange algorithm.
    """
    f = lambda x: -_xlogx(x)  # noqa: E731
    grid = (1 - np.cos(np.pi * np.linspace(0, 1, 40001))) / 2
    fgrid = f(grid)
    n = degree + 2
    ref = (1 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2
    signs = (-1.0) ** np.arange(n)
    for _ in range(100):
        system = np.column_stack([C.chebvander(2 * ref - 1, degree), signs])
        sol = np.linalg.solve(system, f(ref))
        coef, level = sol[:-1], abs(sol[-1])
        err = fgrid - C.chebval(2 * grid - 1, coef)
        # one extremum per run of constant sign
        cuts = np.flatnonzero(np.diff(np.sign(err)) != 0) + 1
        runs = np.split(np.arange(len(grid)), cuts)
        extrema = np.array([run[np.argmax(np.abs(err[run]))] for run in runs])
        while len(extrema) > n:
            drop = 0 if abs(err[extrema[0]]) < abs(err[extrema[-1]]) else -1
            extrema = np.delete(extrema, drop)
        if len(extrema) < n:
            break
        ref = grid[extrema]
        signs = np.sign(err[extrema])
        if np.max(np.abs(err)) - level <= 1e-12 * max(level, 1e-300):
            break
    monomial = C.Chebyshev(coef, domain=[0, 1]).convert(kind=P.Polynomial, domain=[-1, 1], window=[-1, 1]).coef
    return tuple(np.pad(monomial, (0, degree + 1 - len(monomial))))


def _wu_yang(counts, N, K, max_degree=22):
    """
    The minimax-rate polynomial estimator of Wu & Yang: counts above
    :math:`L = \\lfloor 1.6 \\ln K \\rfloor` use the bias-corrected plug-in, and
    the others the unbiased estimate of the best degree-:math:`L` approximation of
    :math:`-p \\ln p` on :math:`[0, 3.5 \\ln K / N]`. The degree is capped at
    `max_degree` to keep the monomial coefficients accurate.
    """
    K = len(counts) if K is None else K
    if K < 2:
        return 0.0
    degree = int(min(max(1, np.floor(1.6 * np.log(K))), max_degree))
    ratio = 3.5 * np.log(K)
    a = np.array(_best_approximation(degree))
    j = np.arange(degree + 1)
    # sum_i a_i (j)_i / M^i: the unbiased (Poisson) estimate of the approximation at p = j M / N
    falling = np.exp(gammaln(j[:, None] + 1) - gammaln(np.maximum(j[:, None] - j[None, :], 0) + 1))
    falling = np.where(j[None, :] <= j[:, None], falling, 0.0)
    g = (falling * a[None, :] / ratio ** j[None, :]).sum(axis=1)
    g = (g * ratio + np.log(N / ratio) * j) / N
    counts = counts.astype(np.int64)
    small = counts <= degree
    p = counts[~small] / N
    h = np.sum(g[counts[small]]) + np.sum(-p * np.log(p) + 0.5 / N) + g[0] * (K - len(counts))
    return max(float(h), 0.0)


def _binomial_matrix(N, rows, p):
    """
    ``P[i, j] = C(N, i) p_j^i (1 - p_j)^(N - i)`` for ``i < rows``.
    """
    i = np.arange(rows)[:, None]
    logc = gammaln(N + 1) - gammaln(i + 1) - gammaln(N - i + 1)
    return np.exp(logc + i * np.log(p)[None, :] + (N - i) * np.log1p(-p)[None, :])


def _bag(N, m, lambda_0=0.0):
    """
    Paninski's BAG coefficients, used by BUB for fewer than 20 samples.
    """
    p = np.linspace(0, 1, 5 * N + 1)
    Pm = _binomial_matrix(N, N + 1, p[1:-1])
    Pm = np.column_stack([np.eye(N + 1)[:, 0], Pm, np.eye(N + 1)[:, -1]])
    X = m * Pm.T
    XY = m * X.T @ -_xlogx(p)
    DD = 2 * np.eye(N + 1) - np.eye(N + 1, k=1) - np.eye(N + 1, k=-1)
    DD[0, 0] = DD[N, N] = 1
    AA = X.T @ X + N * DD
    AA = AA * (np.abs(AA) > np.abs(AA).max() * 1e-7)
    AA[0, 0] += lambda_0
    return pinv(AA) @ XY


def _bub_coefficients(N, m, k_max=11, lambda_0=0.0):
    """
    Paninski's BUB coefficients :math:`a_0, \\ldots, a_N` (nats): the estimate is
    :math:`\\sum_i a_{n_i}` over all :math:`m` outcomes.
    """
    if N < 20:
        return _bag(N, m, lambda_0)
    c = int(np.floor(min(N, 80 * max(N / m, 1))))
    s, mesh = 30, 200
    eps = 1e-10 / N
    p = np.logspace(np.log10(1e-4 / N), np.log10(min(1, s / N) - eps), mesh)
    Pb = _binomial_matrix(N, c + 1, p)
    epsm = 1e-10 / m
    pm = np.arange(epsm, min(1, s / m) - epsm, min(1, s / m) / mesh)
    Pm = _binomial_matrix(N, c + 1, pm)
    f = np.where(pm <= 1 / m, m, 1 / pm)
    target = -_xlogx(p)
    x = np.arange(N + 1) / N
    a = -_xlogx(x) + (1 - x) * 0.5 / N
    mda = np.max(np.abs(np.diff(a)))
    best, best_mm = a.copy(), np.inf
    for k in range(1, min(k_max, N) + 1):
        h_mm = a[k : c + 1] @ Pb[k:]
        XX = m**2 * Pb[:k] @ Pb[:k].T
        XY = m**2 * Pb[:k] @ (target - h_mm)
        XY[k - 1] += N * a[k - 1]
        DD = 2 * np.eye(k) - np.eye(k, k=1) - np.eye(k, k=-1)
        DD[0, 0] = 1
        DD[k - 1, k - 1] = 1
        AA = XX + N * DD
        AA[0, 0] += lambda_0
        AA[k - 1, k - 1] += N
        a[:k] = pinv(AA) @ XY
        bias = m * (a[: c + 1] @ Pb - target)
        V1 = ((np.arange(c + 1) / N) * (a[: c + 1] - np.concatenate([[0], a[:c]])) ** 2) @ Pm
        mmda = max(mda, np.max(np.abs(np.diff(a[: min(k + 2, len(a))]))))
        mm = np.sqrt(np.max(np.abs(bias)) ** 2 + N * min(mmda**2, 4 * np.max(f * V1)))
        if mm < best_mm:
            best, best_mm = a.copy(), mm
    return best


def _bub(counts, N, K, k_max=11):
    """
    Paninski's best-upper-bound estimator: the estimator linear in the count
    histogram that minimizes a bound on the worst-case error over all
    distributions on `K` outcomes.
    """
    K = len(counts) if K is None else K
    a = _bub_coefficients(int(N), K, k_max)
    return float(np.sum(a[counts.astype(np.int64)]) + a[0] * (K - len(counts)))


def _unseen(counts, N, K=None, grid_factor=1.05, alpha=0.5):
    """
    Valiant & Valiant's estimate: recover the histogram of probabilities by
    linear programming from the fingerprint (how many outcomes were seen once,
    twice, ...), then take its entropy.
    """
    N = int(N)
    counts = counts.astype(np.int64)
    finger = np.bincount(counts)[1:].astype(float)
    x_lp_min = 1 / (N * max(10, N))
    seen = np.flatnonzero(finger) + 1
    if seen[0] > 1:
        x_lp_min = seen[0] / N
    x, hist = [], []
    f_lp = np.zeros(len(finger))
    for i in seen:
        window = finger[max(1, i - int(np.ceil(np.sqrt(i)))) - 1 : min(i + int(np.ceil(np.sqrt(i))), len(finger))]
        if window.sum() < np.sqrt(i):
            x.append(i / N)
            hist.append(finger[i - 1])
        else:
            f_lp[i - 1] = finger[i - 1]
    x, hist = np.array(x), np.array(hist)
    if not f_lp.any():
        return float(-np.sum(hist * _xlogx(x)))
    f_max = np.flatnonzero(f_lp)[-1] + 1
    lp_mass = 1 - np.sum(x * hist)
    f_lp = np.concatenate([f_lp[:f_max], np.zeros(int(np.ceil(np.sqrt(f_max))))])
    n_f = len(f_lp)
    x_lp = x_lp_min * grid_factor ** np.arange(int(np.ceil(np.log(f_max / N / x_lp_min) / np.log(grid_factor))) + 1)
    n_x = len(x_lp)
    # variables: mass at each grid point (h(x) x), then a slack pair per fingerprint entry
    pois = poisson.pmf(np.arange(1, n_f + 1)[:, None], N * x_lp[None, :]) / x_lp[None, :]
    A = np.zeros((2 * n_f, n_x + 2 * n_f))
    A[0::2, :n_x], A[1::2, :n_x] = pois, -pois
    A[np.arange(0, 2 * n_f, 2), n_x + np.arange(0, 2 * n_f, 2)] = -1
    A[np.arange(1, 2 * n_f, 2), n_x + np.arange(1, 2 * n_f, 2)] = -1
    b = np.empty(2 * n_f)
    b[0::2], b[1::2] = f_lp, -f_lp
    objective = np.zeros(n_x + 2 * n_f)
    objective[n_x:] = np.repeat(1 / np.sqrt(f_lp + 1), 2)
    A_eq = np.zeros((1, n_x + 2 * n_f))
    A_eq[0, :n_x] = 1
    first = linprog(objective, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[lp_mass], method="highs")
    support = np.zeros(n_x + 2 * n_f)
    support[:n_x] = 1 / x_lp
    second = linprog(
        support,
        A_ub=np.vstack([A, objective]),
        b_ub=np.concatenate([b, [first.fun + alpha]]),
        A_eq=A_eq,
        b_eq=[lp_mass],
        method="highs",
    )
    hist_lp = second.x[:n_x] / x_lp
    return float(-np.sum(hist * _xlogx(x)) - np.sum(hist_lp * _xlogx(x_lp)))


def _pym_gamma(alpha, d):
    return (digamma(1) - digamma(1 - d)) / (digamma(alpha + 1) - digamma(1 - d))


def _pym_log_prior(alpha, d, param=0.1):
    return -1 / ((1 - _pym_gamma(alpha, d)) * param)


def _pym_log_likelihood(alpha, d, values, mult, N, K):
    """
    The log probability of the observed table occupancies under a Pitman-Yor process.
    """
    alpha, d = np.atleast_1d(alpha)[:, None], np.atleast_1d(d)[:, None]
    k = np.arange(1, K)[None, :]
    return (
        -(gammaln(alpha + N) - gammaln(alpha + 1)).ravel()
        + np.log(alpha + k * d).sum(axis=1)
        + (mult[None, :] * (gammaln(values[None, :] - d) - gammaln(1 - d))).sum(axis=1)
    )


def _pym_conditional_mean(alpha, d, values, mult, N, K):
    """
    The posterior mean entropy (nats) given the Pitman-Yor parameters.
    """
    alpha, d = np.asarray(alpha, dtype=float), np.asarray(d, dtype=float)
    a_kd = alpha + K * d
    prior_mean = digamma(1 + a_kd) - digamma(1 - d)
    tail = (
        digamma(alpha + N + 1)
        - a_kd / (alpha + N) * digamma(a_kd + 1)
        - (N - K * d) / (alpha + N) * digamma(N - K * d + 1)
    )
    shifted = values[None, :] - d[:, None]
    seen = digamma(N - K * d + 1) - (mult[None, :] * shifted / (N - K * d)[:, None] * digamma(1 + shifted)).sum(axis=1)
    return (N - K * d) / (alpha + N) * seen + a_kd / (alpha + N) * prior_mean + tail


def _pym(counts, N, K=None, n_grid=25):
    """
    The Pitman-Yor mixture estimator of Archer, Park & Pillow: the posterior mean
    entropy under a mixture of Pitman-Yor process priors, integrated over the
    concentration and discount by quadrature around the posterior mode. It
    assumes a countably infinite alphabet, so `K` is ignored.
    """
    values, mult = np.unique(counts.astype(np.int64), return_counts=True)
    values = values.astype(float)
    mult = mult.astype(float)
    K_obs = len(counts)
    if values.max() <= 1:
        raise ValueError("The PYM estimator needs at least one repeated outcome.")
    if K_obs == 1:
        raise ValueError("The PYM estimator needs at least two distinct outcomes.")
    eps = np.finfo(float).eps

    def neg_log_post(x):
        alpha, d = x
        if alpha <= 0 or d <= 0 or d >= 1:
            return np.inf
        return -(_pym_log_likelihood(alpha, d, values, mult, N, K_obs)[0] + _pym_log_prior(alpha, d))

    start = minimize(neg_log_post, [1.0, 0.01], method="L-BFGS-B", bounds=[(1e-10, None), (1e-10, 1 - 1e-10)])
    mode = start.x
    h = np.maximum(1e-4 * np.abs(mode), 1e-7)
    hess = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
            hess[i, j] = (
                neg_log_post(mode + ei + ej)
                - neg_log_post(mode + ei - ej)
                - neg_log_post(mode - ei + ej)
                + neg_log_post(mode - ei - ej)
            ) / (4 * h[i] * h[j])
    try:
        ok = np.all(np.isfinite(hess)) and np.linalg.cholesky(hess) is not None
    except np.linalg.LinAlgError:
        ok = False
    nodes, weights = legendre.leggauss(n_grid)

    def gauss(lo, hi):
        return (hi - lo) / 2 * nodes + (hi + lo) / 2, (hi - lo) / 2 * weights

    if ok:
        spread = 6 * np.sqrt(np.diag(np.linalg.inv(hess)))
        a_x, a_w = gauss(max(mode[0] - spread[0], eps), mode[0] + spread[0])
        d_x, d_w = gauss(max(mode[1] - spread[1], eps), min(1 - eps, max(mode[1] + spread[1], eps)))
    else:
        # Boyd's rational Chebyshev rule on the semi-infinite alpha axis
        n_a = 10 * n_grid
        t = np.pi * np.arange(1, n_a + 1) / (n_a + 1)
        j = np.arange(1, n_a + 1)
        a_x = 1 / np.tan(t / 2) ** 2
        a_w = np.sin(np.outer(j, t)).T @ ((1 - np.cos(np.pi * j)) / j)
        a_w = 2 * np.sin(t) / (1 - np.cos(t)) ** 2 * a_w * 2 / (n_a + 1)
        d_x, d_w = gauss(eps, 1 - eps)
    aa, dd = np.meshgrid(a_x, d_x)
    aa, dd = aa.ravel(), dd.ravel()
    log_lik = _pym_log_likelihood(aa, dd, values, mult, N, K_obs)
    log_w = log_lik + _pym_log_prior(aa, dd)
    w = np.exp(log_w - log_w.max()) * np.outer(d_w, a_w).ravel()
    return float(np.sum(w * _pym_conditional_mean(aa, dd, values, mult, N, K_obs)) / np.sum(w))
