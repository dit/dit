"""
Channel capacity via the Blahut-Arimoto algorithm.

The iteration follows :cite:`Blahut1972,Arimoto1972` (see also
:cite:`Cover2006`, Section 10.8). Every iterate ``r`` of the input distribution
gives the certified bounds

    I(r) = sum_x r(x) D(P_x || rP)  <=  C  <=  max_x D(P_x || rP),

so the loop stops once the gap between them is below tolerance.
"""

import warnings

import numpy as np

import dit

from ..cdisthelpers import cdist_array
from ..exceptions import ditException
from ..utils import unitful

__all__ = (
    "channel_capacity",
    "channel_capacity_cost",
    "sibson_capacity",
)

_LN2 = np.log(2)
_TINY = np.finfo(float).tiny
_MAX_STEP = 1e6


def _row_plogp(P):
    """
    Compute ``sum_y P(y|x) ln P(y|x)`` for each row, with ``0 ln 0 = 0``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(P > 0, P * np.log(P), 0.0).sum(axis=1)


def _divergences(P, plogp, q):
    """
    Compute ``D(P_x || q)`` in nats for every row ``x`` of ``P``.
    """
    return plogp - P @ np.log(np.maximum(q, _TINY))


def _normalize_log(logr):
    """
    Normalize log-weights and return ``(log r, r)``.
    """
    logr = logr - logr.max()
    r = np.exp(logr)
    total = r.sum()
    return logr - np.log(total), r / total


def _kl(a, b):
    """
    ``D(a || b)`` in nats.
    """
    mask = a > 0
    return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], _TINY))))


def _blahut_arimoto(P, plogp, logr, tol, max_iters, accelerated, offset=None):
    """
    Iterate Blahut-Arimoto on the channel ``P`` from the log-input ``logr``.

    Parameters
    ----------
    P : np.ndarray
        Row-stochastic channel matrix.
    plogp : np.ndarray
        ``_row_plogp(P)``.
    logr : np.ndarray
        Initial (normalized) log input distribution.
    tol : callable
        Maps the current lower bound (nats) to the allowed bound gap (nats).
    max_iters : int
        Maximum number of iterations.
    accelerated : bool
        Use the accelerated update ``r <- r exp(mu D)`` of
        :cite:`matz2004information` with the adaptive step
        ``mu = D(r_k || r_{k-1}) / D(q_k || q_{k-1}) >= 1``. Whenever that step
        would decrease the lower bound, the plain step ``mu = 1``, which never
        does, is taken instead.
    offset : np.ndarray, None
        Per-input offset added to the divergences, e.g. ``-s * cost`` for the
        cost-constrained capacity. The bounds then refer to the Lagrangian.

    Returns
    -------
    logr, r, q, D : np.ndarray
        Final log input, input, output distributions and (offset) divergences.
    lower, upper : float
        Bounds on the (Lagrangian) capacity in nats.
    iters : int
        Iterations used.
    """
    logr, r = _normalize_log(logr)

    def evaluate(r):
        q = r @ P
        D = _divergences(P, plogp, q)
        if offset is not None:
            D = D + offset
        return q, D

    q, D = evaluate(r)
    lower = float(r @ D)
    upper = float(D.max())
    mu = 1.0
    iters = 0
    while upper - lower > tol(lower) and iters < max_iters:
        iters += 1
        new_logr, new_r = _normalize_log(logr + mu * D)
        new_q, new_D = evaluate(new_r)
        new_lower = float(new_r @ new_D)
        if accelerated:
            if mu > 1.0 and new_lower < lower:
                mu = 1.0
                new_logr, new_r = _normalize_log(logr + D)
                new_q, new_D = evaluate(new_r)
                new_lower = float(new_r @ new_D)
            else:
                dq = _kl(new_q, q)
                mu = min(_kl(new_r, r) / dq, _MAX_STEP) if dq > 0 else 1.0
        logr, r, q, D, lower = new_logr, new_r, new_q, new_D, new_lower
        upper = float(D.max())
    return logr, r, q, D, lower, upper, iters


def _active_set(P, plogp, tol, max_iters, accelerated, offset=None):
    """
    Blahut-Arimoto restricted to a growing set of inputs :cite:`sayir2000iterating`.

    The sub-channel on the active inputs is solved to tolerance; the iteration
    stops once every inactive input also satisfies the Kuhn-Tucker condition
    ``D(P_x || q) <= lower + tol``, which is exactly Blahut's bound gap
    :cite:`Blahut1972` over the full input alphabet.
    """
    n = P.shape[0]
    uniform = np.full(n, 1 / n)
    D = _divergences(P, plogp, uniform @ P)
    if offset is not None:
        D = D + offset
    k = min(n, 2)
    active = np.sort(np.argpartition(-D, k - 1)[:k])
    logr = np.zeros(len(active))
    total_iters = 0
    while True:
        sub_offset = None if offset is None else offset[active]
        logr, r, q, _, lower, upper, iters = _blahut_arimoto(
            P[active],
            plogp[active],
            logr,
            tol,
            max_iters - total_iters,
            accelerated,
            sub_offset,
        )
        total_iters += iters
        D = _divergences(P, plogp, q)
        if offset is not None:
            D = D + offset
        upper = max(upper, float(D.max()))
        if upper - lower <= tol(lower) or len(active) == n or total_iters >= max_iters:
            break
        inactive = np.setdiff1d(np.arange(n), active)
        violators = inactive[D[inactive] > lower + tol(lower)]
        if len(violators) == 0:
            break
        violators = violators[np.argsort(-D[violators])][: len(active)]
        logr = np.concatenate([logr, np.full(len(violators), logr.max())])
        order = np.argsort(np.concatenate([active, violators]))
        active = np.concatenate([active, violators])[order]
        logr = logr[order]
    r_full = np.zeros(n)
    r_full[active] = r
    return r_full, lower, upper, total_iters


def _weakly_symmetric(P):
    """
    Closed form for weakly symmetric channels (:cite:`Cover2006`, Theorem 7.2.1).

    Returns ``(capacity in nats, uniform input)`` or ``None``.
    """
    n, m = P.shape
    sorted_rows = np.sort(P, axis=1)
    if not np.allclose(sorted_rows, sorted_rows[0], rtol=0, atol=1e-14):
        return None
    col_sums = P.sum(axis=0)
    if not np.allclose(col_sums, col_sums[0], rtol=0, atol=1e-12):
        return None
    row = P[0]
    entropy = -float(np.sum(row[row > 0] * np.log(row[row > 0])))
    return np.log(m) - entropy, np.full(n, 1 / n)


def _square_invertible(P, plogp):
    """
    Closed form for a square, nonsingular channel matrix.

    If every input is used, ``D(P_x || q) = C`` for all ``x``, i.e.
    ``sum_y P(y|x) ln q(y) = plogp(x) - C``. For nonsingular ``P`` this is
    solved by ``P c = plogp``, ``C = ln sum_y exp(c_y)`` and
    ``q_y = exp(c_y - C)``, with input ``p = q P^{-1}``, valid when ``p >= 0``.

    Returns ``(input distribution)`` or ``None``; the caller certifies it.
    """
    n, m = P.shape
    if n != m or np.linalg.cond(P) > 1e10:
        return None
    try:
        c = np.linalg.solve(P, plogp)
    except np.linalg.LinAlgError:
        return None
    c_max = c.max()
    q = np.exp(c - c_max)
    q /= q.sum()
    p = np.linalg.solve(P.T, q)
    if np.any(p < -1e-12):
        return None
    p = np.clip(p, 0, None)
    return p / p.sum()


def _reduce_channel(P):
    """
    Drop all-zero output columns and collapse duplicate or empty input rows.

    Returns ``(P_reduced, keep)`` where ``keep[i]`` is the original row index
    of reduced row ``i``.
    """
    valid = np.flatnonzero(np.isfinite(P).all(axis=1) & (P.sum(axis=1) > 0))
    P = P[:, P[valid].sum(axis=0) > 0]
    _, first = np.unique(P[valid], axis=0, return_index=True)
    keep = valid[np.sort(first)]
    return P[keep], keep


def _tolerance(rtol, atol):
    """
    Allowed bound gap in nats as a function of the lower bound in nats.
    """
    if rtol is None:
        rtol = dit.ditParams["rtol"]
    if atol is None:
        atol = dit.ditParams["atol"]

    def tol(lower):
        return max(atol, rtol * abs(lower) / _LN2) * _LN2

    return tol


def _warn_unconverged(name, max_iters, lower, upper):
    msg = f"{name} did not reach tolerance in {max_iters} iterations; the value is in [{lower}, {upper}] bits."
    warnings.warn(msg, RuntimeWarning, stacklevel=4)


def _capacity(P, rtol, atol, max_iters, method, eliminate):
    """
    Compute the capacity (bits) and an optimal input pmf of the channel ``P``.
    """
    if method not in ("ba", "accelerated"):
        msg = f"Unknown method {method!r}; expected 'ba' or 'accelerated'."
        raise ditException(msg)
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    tol = _tolerance(rtol, atol)

    closed = _weakly_symmetric(P)
    if closed is not None:
        cc, pmf = closed
        return cc / _LN2, pmf

    R, keep = _reduce_channel(P)
    plogp = _row_plogp(R)
    pmf = np.zeros(n)

    r = _square_invertible(R, plogp)
    if r is not None:
        D = _divergences(R, plogp, r @ R)
        lower, upper = float(r @ D), float(D.max())
        if upper - lower <= tol(lower):
            pmf[keep] = r
            return lower / _LN2, pmf

    accelerated = method == "accelerated"
    if eliminate and len(R) > 2:
        r, lower, upper, _ = _active_set(R, plogp, tol, max_iters, accelerated)
    else:
        logr = np.full(len(R), -np.log(len(R)))
        _, r, _, _, lower, upper, _ = _blahut_arimoto(R, plogp, logr, tol, max_iters, accelerated)
    if upper - lower > tol(lower):
        _warn_unconverged("Blahut-Arimoto", max_iters, lower / _LN2, upper / _LN2)
    pmf[keep] = r
    return lower / _LN2, pmf


def _channel_array(cdists):
    """
    Convert the accepted channel representations to a dense 2D array.
    """
    from ..distribution import Distribution

    is_xr = isinstance(cdists, Distribution)
    is_xr_conditional = is_xr and cdists.is_conditional()

    if is_xr and not is_xr_conditional:
        raise ditException("Distribution passed to channel_capacity must be conditional")

    if is_xr_conditional:
        lin = cdists._linear_data()
        given_dims = [d for d in cdists.dims if d in cdists.given_vars]
        free_dims = [d for d in cdists.dims if d in cdists.free_vars]
        reordered = lin.transpose(*given_dims, *free_dims)
        n_given = int(np.prod([len(cdists.data.coords[d]) for d in given_dims]))
        n_free = int(np.prod([len(cdists.data.coords[d]) for d in free_dims]))
        return reordered.values.reshape(n_given, n_free), True

    try:
        cdists.shape  # noqa: B018
        return np.asarray(cdists), False
    except AttributeError:
        return cdist_array(cdists, base="linear", mode="dense"), False


def channel_capacity(
    cdists,
    marginal=None,
    rtol=None,
    atol=None,
    max_iters=100_000,
    method="accelerated",
    eliminate=False,
):
    """
    Calculates the channel capacity from conditional distributions P(Y|X).

    The capacity is computed with the Blahut-Arimoto algorithm
    :cite:`Blahut1972,Arimoto1972`, stopping once Blahut's lower and upper
    bounds on the capacity agree to within tolerance, so the returned value is
    within ``max(atol, rtol * C)`` bits of the true capacity. Weakly symmetric
    :cite:`Cover2006` and square nonsingular channels are
    solved in closed form when the closed form certifies.

    Parameters
    ----------
    cdists : list, ndarray, Distribution
        A list of conditional distributions, a 2D array where rows are
        inputs and columns are outputs, or a conditional Distribution
        such as p(Y|X).
    marginal : distribution | None
        The marginal distribution P(X). If provided, P*(X) is returned as
        a distribution object. If None, P*(X) is returned as a numpy array.
    rtol : float or None
        Relative tolerance on the capacity. Defaults to ditParams['rtol'].
    atol : float or None
        Absolute tolerance on the capacity, in bits. Defaults to
        ditParams['atol'].
    max_iters : int
        Maximum number of Blahut-Arimoto iterations. A ``RuntimeWarning`` is
        raised if the bounds have not met by then.
    method : {'accelerated', 'ba'}
        ``'ba'`` is the classic iteration; ``'accelerated'`` takes the step
        ``r <- r exp(mu D)`` of :cite:`matz2004information` with the adaptive
        ``mu = D(r_k || r_{k-1}) / D(q_k || q_{k-1})``, falling back to the
        classic step whenever the mutual information would decrease.
    eliminate : bool
        If True, iterate on a growing active set of inputs
        :cite:`sayir2000iterating`. This helps when only a few of many inputs
        carry probability at capacity.

    Returns
    -------
    cc : float
        The channel capacity.
    marginal_opt : distribution or ndarray
        The optimal marginal P*(X).
    """
    from ..distribution import Distribution

    carr, is_xr_conditional = _channel_array(cdists)

    if marginal is not None:
        n_inputs = len(marginal)
        if n_inputs != carr.shape[0]:
            msg = "len(mdist) != len(cdists)"
            raise ditException(msg)

    cc, pmf = _capacity(carr, rtol, atol, max_iters, method, eliminate)

    if marginal is not None:
        if is_xr_conditional and isinstance(marginal, Distribution):
            result = marginal.copy()
            result.data = result.data.copy(deep=True)
            result.data.values[:] = pmf.reshape(result.data.shape)
            marginal_opt = result
        else:
            marginal_opt = marginal.copy()
            marginal_opt.pmf = pmf
    else:
        marginal_opt = pmf

    return cc, marginal_opt


@unitful
def channel_capacity_joint(dist, input_, output, marginal=False):
    """
    Compute the channel capacity from ``input_`` to ``output``.

    Parameters
    ----------
    dist : Distribution
        The joint distribution.
    input_ : iterable
        The random variables that are the input of the channel.
    output : iterable
        The random variables that are the output of the channel.
    marginal : bool
        Whether to return the marginal distribution. Defaults to False.
    """
    input_names = list(dist._resolve_rv_names(list(input_)))
    output_names = list(dist._resolve_rv_names(list(output)))

    keep_vars = input_names + output_names
    sub = dist.marginal(*keep_vars)
    marg, cdist_list = sub.condition_on(input_names)
    cc, marg_opt = channel_capacity(cdist_list, marg)
    if marginal:
        return cc, marg_opt
    else:
        return cc


def channel_capacity_cost(cdists, cost, budget, rtol=None, atol=None, max_iters=100_000, method="accelerated"):
    """
    Capacity of a channel whose inputs carry a cost, under an average budget.

    Computes ``C(B) = max { I[X : Y] : E[cost(X)] <= B }`` through Blahut's
    Lagrangian form :cite:`Blahut1972`: for a multiplier ``s >= 0`` the
    iteration maximizes ``I[X : Y] - s E[cost(X)]`` (the offset divergences
    ``D(P_x || q) - s cost(x)`` keep the bound-gap certificate), and
    ``C(B) = min_s max_p I - s (E[cost] - B)``. The multiplier is found by
    root-finding ``E[cost] = B``.

    Parameters
    ----------
    cdists : list, ndarray, Distribution
        The channel, in any form accepted by :func:`channel_capacity`.
    cost : array_like
        The cost of each input, in the same order as the channel's rows.
    budget : float
        The maximum average cost.
    rtol : float or None
        Relative tolerance. Defaults to ditParams['rtol'].
    atol : float or None
        Absolute tolerance in bits. Defaults to ditParams['atol'].
    max_iters : int
        Maximum Blahut-Arimoto iterations per value of the multiplier.
    method : {'accelerated', 'ba'}
        As in :func:`channel_capacity`.

    Returns
    -------
    cc : float
        The cost-constrained capacity, in bits.
    pmf : np.ndarray
        An input distribution achieving it. If the optimal input at the final
        multiplier is not unique its cost may differ slightly from ``budget``.
    s : float
        The Lagrange multiplier, in bits per unit cost; the slope of ``C(B)``.
        ``0`` when the budget does not bind and ``inf`` when only the cheapest
        inputs are affordable.

    Raises
    ------
    ditException
        If the budget is below the cheapest input's cost.
    """
    if method not in ("ba", "accelerated"):
        msg = f"Unknown method {method!r}; expected 'ba' or 'accelerated'."
        raise ditException(msg)
    P, _ = _channel_array(cdists)
    P = np.asarray(P, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (P.shape[0],):
        msg = "cost must have one entry per channel input."
        raise ditException(msg)
    budget = float(budget)
    c_min = float(cost.min())
    if budget < c_min - 1e-12:
        msg = f"Budget {budget} is below the minimum input cost {c_min}."
        raise ditException(msg)

    n = P.shape[0]
    cheapest = np.flatnonzero(cost <= c_min + 1e-12)
    if budget <= c_min + 1e-12:
        cc, sub = _capacity(P[cheapest], rtol, atol, max_iters, method, False)
        pmf = np.zeros(n)
        pmf[cheapest] = sub
        return cc, pmf, np.inf

    from scipy.optimize import brentq

    tol = _tolerance(rtol, atol)
    accelerated = method == "accelerated"
    R = P[:, P.sum(axis=0) > 0]
    plogp = _row_plogp(R)
    state = {"logr": np.full(n, -np.log(n))}

    def solve(s):
        offset = None if s == 0 else -s * _LN2 * cost
        logr, r, _, _, lower, upper, _ = _blahut_arimoto(R, plogp, state["logr"], tol, max_iters, accelerated, offset)
        state["logr"] = logr
        if upper - lower > tol(lower):
            _warn_unconverged("Blahut-Arimoto", max_iters, lower / _LN2, upper / _LN2)
        return r, lower / _LN2

    r, lagrangian = solve(0.0)
    if r @ cost <= budget:
        return lagrangian, r, 0.0

    def excess(s):
        return solve(s)[0] @ cost - budget

    s_hi = 1.0
    while excess(s_hi) > 0:
        s_hi *= 2
    xtol = tol(0.0) / _LN2 / max(float(cost.max()) - c_min, 1.0)
    s = brentq(excess, 0.0, s_hi, xtol=xtol)
    r, lagrangian = solve(s)
    return lagrangian + s * budget, r, s


def sibson_capacity(cdists, order, rtol=None, atol=None, max_iters=100_000):
    """
    Sibson's capacity of order alpha, ``C_alpha = max_p I_alpha(X : Y)``.

    Sibson's alpha-mutual information is

        I_alpha(X : Y) = alpha / (alpha - 1) log2 sum_y (sum_x p(x) P(y|x)^alpha)^(1/alpha),

    and its maximum over inputs equals the Rényi radius
    ``min_q max_x D_alpha(P_x || q)`` :cite:`csiszar1995generalized`. Every
    input ``r`` therefore gives the bounds ``I_alpha(r) <= C_alpha <=
    max_x D_alpha(P_x || q_r)``, where ``q_r`` is Sibson's optimal output for
    ``r``. The iteration ``r <- r exp(mu D_alpha(P_x || q_r))`` with
    ``mu = max(1, alpha)`` reduces to Blahut-Arimoto at ``alpha = 1``; the step
    is halved whenever ``I_alpha`` would decrease, and the iteration stops once
    the bounds meet.

    Parameters
    ----------
    cdists : list, ndarray, Distribution
        The channel, in any form accepted by :func:`channel_capacity`.
    order : float
        The order ``alpha > 0``. ``1`` gives the Shannon capacity and
        ``numpy.inf`` the maximal leakage capacity ``log2 sum_y max_x P(y|x)``.
    rtol : float or None
        Relative tolerance. Defaults to ditParams['rtol'].
    atol : float or None
        Absolute tolerance in bits. Defaults to ditParams['atol'].
    max_iters : int
        Maximum number of iterations.

    Returns
    -------
    cc : float
        The capacity of order alpha, in bits.
    pmf : np.ndarray
        An optimal input distribution.
    """
    from scipy.special import logsumexp

    order = float(order)
    if not order > 0:
        msg = "order must be positive."
        raise ditException(msg)
    P, _ = _channel_array(cdists)
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if order == 1:
        return channel_capacity(P, rtol=rtol, atol=atol, max_iters=max_iters)
    P = P[:, P.sum(axis=0) > 0]
    if np.isinf(order):
        return float(np.log2(P.max(axis=0).sum())), np.full(n, 1 / n)

    a = order
    tol = _tolerance(rtol, atol)
    with np.errstate(divide="ignore"):
        logP = np.log(P)

    def evaluate(logr):
        log_g = logsumexp(logr[:, np.newaxis] + a * logP, axis=0)
        log_S = logsumexp(log_g / a)
        logq = log_g / a - log_S
        D = logsumexp(a * logP + (1 - a) * logq, axis=1) / (a - 1)
        return D, a / (a - 1) * log_S

    logr, _ = _normalize_log(np.zeros(n))
    D, lower = evaluate(logr)
    upper = float(D.max())
    mu = max(1.0, a)
    iters = 0
    while upper - lower > tol(lower) and iters < max_iters:
        iters += 1
        slack = 1e-3 * (upper - lower) + 64 * np.finfo(float).eps * abs(lower)
        while True:
            new_logr, _ = _normalize_log(logr + mu * (D - upper))
            new_D, new_lower = evaluate(new_logr)
            if new_lower >= lower - slack or mu < 1e-12:
                break
            mu /= 2
        if new_lower < lower - slack:
            break
        logr, D, lower = new_logr, new_D, new_lower
        upper = float(D.max())
    if upper - lower > tol(lower):
        _warn_unconverged("sibson_capacity", max_iters, lower / _LN2, upper / _LN2)
    return lower / _LN2, np.exp(logr)
