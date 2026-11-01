"""
The more-capable intersection information I_mc^∩ from Gomes & Figueiredo (2023).

Defines redundancy as:

    I_mc^∩(Y1, ..., Yn → T) = max_{Q : Q ≤_mc Y_i ∀i} I(Q; T)

where ≤_mc is the "more capable" channel preorder: K_Q ≤_mc K^(i) iff
I(Q; T) ≤ I(Y_i; T) for *every* input distribution P(T).

The more-capable constraint set involves an uncountable family of
inequalities. Following the paper (Section 5.2, eq. 13), we discretize the
input simplex with S sampled distributions and search the resulting
finite-constraint problem with SLSQP from several starts, keeping the best
candidate that satisfies the sampled constraints. Every common degradation of
the sources is less capable than each of them, so the optimum of `ideg` is
one of the candidates and the result is at least I_d^cap. The search is local
and the constraints are sampled, so the result is neither a certified lower
nor upper bound on I_mc^cap.

References
----------
.. [1] A. F. C. Gomes and M. A. T. Figueiredo, "Orders between Channels
       and Implications for Partial Information Decomposition",
       Entropy 25, 975, 2023.
.. [2] A. Kolchinsky, "A Novel Approach to the Partial Information
       Decomposition", Entropy 24, 403, 2022.
"""

import numpy as np
from scipy.optimize import minimize

from ...shannon import mutual_information
from ...utils.deprecation import renamed_kwarg
from ..pid import BasePID
from .ideg import _degradation_lp, _source_channels

__all__ = ("PID_MC",)


def _params_to_stochastic(params, n_rows, n_cols):
    """Map unconstrained params to a row-stochastic matrix via softmax."""
    mat = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        logits = params[r * n_cols : (r + 1) * n_cols]
        x = logits - logits.max()
        e = np.exp(x)
        mat[r] = e / e.sum()
    return mat


def _mi_bits(pi_t, ch):
    """I(T; output) in bits for channel ch = P(output|T) given pi_t."""
    return float(_batch_mi(pi_t[None, :], ch)[0])


def _batch_mi(pis, ch):
    """
    I(T; output) in bits for the channel ch = P(output|T), for each input distribution (row)
    of `pis`.
    """
    joint = pis[:, :, None] * ch[None, :, :]
    p_out = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 1e-300, joint * np.log2(ch[None, :, :] / p_out), 0.0)
    return terms.sum(axis=(1, 2))


def _batch_mi_grad_wrt_K(pis, ch):
    """
    Gradient of :func:`_batch_mi` w.r.t. the channel, one (|T|, |Q|) slice per row of `pis`.
    """
    p_out = pis @ ch
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(ch[None, :, :] / (p_out[:, None, :] + 1e-300))
    return np.where(pis[:, :, None] > 0, pis[:, :, None] * log_ratio, 0.0)


def _mi_grad_wrt_K(pi_t, ch):
    """
    Gradient of :func:`_mi_bits` w.r.t. the channel ``ch``.

    The ``-p log p`` terms cancel, leaving ``dI/dK[t, q] = pi[t] * log2(K[t,q]/p[q])``.
    eps-guarded consistently with :func:`_mi_bits` (denominator only); ``ch`` from
    softmax is strictly positive so the numerator never hits ``log2(0)``.
    """
    eps = 1e-300
    p_out = pi_t @ ch
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(ch / (p_out[None, :] + eps))
    # Rows with pi[t] == 0 contribute nothing (and avoid 0 * -inf -> nan).
    return np.where(pi_t[:, None] > 0, pi_t[:, None] * log_ratio, 0.0)


def _softmax_vjp(ch, g):
    """
    Map an upstream gradient ``g = dObj/dK`` to the gradient w.r.t. the softmax
    logits that produced ``ch`` (row-wise softmax via :func:`_params_to_stochastic`).

    For ``y = softmax(z)``: ``dObj/dz = y * (g - sum_c g_c y_c)`` per row.
    """
    inner = (g * ch).sum(axis=1, keepdims=True)
    return (ch * (g - inner)).ravel()


def _sample_simplex(n_dim, n_samples, rng):
    """Draw n_samples points uniformly from the (n_dim-1)-simplex."""
    raw = rng.exponential(size=(n_samples, n_dim))
    return raw / raw.sum(axis=1, keepdims=True)


def _constraint_distributions(n_t, n_interior, rng):
    """
    Build a set of input distributions that cover the constraint surface.

    Includes interior simplex points, pairwise edge distributions (critical
    for detecting channels that must have identical rows), and the actual
    simplex vertices.
    """
    pts = []

    # Interior simplex samples
    if n_interior > 0:
        pts.append(_sample_simplex(n_t, n_interior, rng))

    # Pairwise edge distributions: for each pair (i,j), distributions
    # supported on only {i, j} at several mixture ratios.  These are the
    # critical constraints that force K_Q rows to agree (Theorem 3 in the
    # paper) and live on measure-zero faces of the simplex.
    edge_alphas = np.linspace(0.1, 0.9, 5)
    for i in range(n_t):
        for j in range(i + 1, n_t):
            for a in edge_alphas:
                p = np.zeros(n_t)
                p[i] = a
                p[j] = 1.0 - a
                pts.append(p.reshape(1, -1))

    # Vertices (trivially satisfied but cheap)
    pts.append(np.eye(n_t))

    return np.vstack(pts)


def _more_capable_ii(channels, pi_t, n_q=None, n_samples=None, niter=None, seed=None):
    """
    Compute I_mc^∩ via discretized more-capable constraints.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T) (used as the actual MI to maximize).
    n_q : int or None
        Cardinality of Q.  Defaults to max(|Y_i|).
    n_samples : int or None
        Number of sampled input distributions for the constraints.
    niter : int or None
        Number of random restarts.
    seed : int or None
        Random seed for reproducibility of the simplex samples.

    Returns
    -------
    float
        The largest I(Q; T) found among candidates satisfying the sampled
        constraints (in bits).
    """
    n_t = channels[0].shape[0]
    sizes = [ch.shape[1] for ch in channels]

    if n_q is None:
        n_q = max(sizes)
    if n_samples is None:
        n_samples = max(50, 10 * n_t)
    if niter is None:
        niter = 20

    rng = np.random.default_rng(seed)
    sampled_pi = _constraint_distributions(n_t, n_samples, rng)
    sampled_pi = np.vstack([sampled_pi, pi_t.reshape(1, -1)])

    # min_i I(Y_i; T) for each sampled distribution
    mi_bounds = np.min([_batch_mi(sampled_pi, ch) for ch in channels], axis=0)

    atol = 1e-6
    best_mi = 0.0

    def _check_and_update(k_q):
        nonlocal best_mi
        if np.all(_batch_mi(sampled_pi, k_q) - mi_bounds <= atol):
            best_mi = max(best_mi, _mi_bits(pi_t, k_q))

    # Phase 1: each source channel, and an optimal common degradation, which is
    # less capable than every source; so the result is at least I_d^cap
    for ch in channels:
        _check_and_update(ch)
    _, lambdas = _degradation_lp(channels, pi_t)
    k_deg = channels[0] @ lambdas[0]
    _check_and_update(k_deg)

    # Phase 2: SLSQP with the sampled inequality constraints from multiple starts
    n_params = n_t * n_q

    def _objective(params):
        k_q = _params_to_stochastic(params, n_t, n_q)
        return -_mi_bits(pi_t, k_q)

    def _objective_jac(params):
        k_q = _params_to_stochastic(params, n_t, n_q)
        return _softmax_vjp(k_q, -_mi_grad_wrt_K(pi_t, k_q))

    def _constraint(params):
        k_q = _params_to_stochastic(params, n_t, n_q)
        return mi_bounds - _batch_mi(sampled_pi, k_q)

    def _constraint_jac(params):
        k_q = _params_to_stochastic(params, n_t, n_q)
        g = -_batch_mi_grad_wrt_K(sampled_pi, k_q)
        inner = (g * k_q[None]).sum(axis=2, keepdims=True)
        return (k_q[None] * (g - inner)).reshape(len(sampled_pi), -1)

    constraints = [{"type": "ineq", "fun": _constraint, "jac": _constraint_jac}]

    # Initial points: near each source channel, the common degradation, plus random
    init_points = []
    for ch in [*channels, k_deg]:
        if ch.shape[1] <= n_q:
            padded = np.full((n_t, n_q), 1e-10)
            padded[:, : ch.shape[1]] = ch
            init_points.append(np.log(padded + 1e-10).ravel())
    for _ in range(max(1, niter - len(init_points))):
        init_points.append(rng.standard_normal(n_params) * 0.5)

    for x0 in init_points:
        try:
            res = minimize(
                _objective,
                x0,
                method="SLSQP",
                jac=_objective_jac,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-12},
            )
            _check_and_update(_params_to_stochastic(res.x, n_t, n_q))
        except Exception:  # pragma: no cover
            continue

    return best_mi


class PID_MC(BasePID):
    """
    More-capable intersection information I_mc^∩ from Gomes & Figueiredo.

    Redundancy is the maximum I(Q; T) such that K_Q is "less capable"
    than every source channel for all input distributions. Defined for any
    number of sources.

    The more-capable constraints are imposed at sampled input distributions and
    the maximum is searched locally, from several starts that include an
    optimal common degradation of the sources. The result is therefore at least
    the degradation redundancy of :class:`~dit.pid.PID_Deg`, but is not a
    certified bound on I_mc^cap. The search is seeded by `rng` (0 by default),
    so results are reproducible.

    References
    ----------
    A. F. C. Gomes and M. A. T. Figueiredo, "Orders between
    Channels and Implications for Partial Information
    Decomposition", Entropy 25, 975, 2023.
    """

    _name = "I_mc∩"

    @staticmethod
    @renamed_kwarg("seed", "rng")
    def _measure(d, sources, target, n_samples=None, niter=None, rng=0):
        """
        Compute the more-capable II redundancy of some sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        n_samples : int or None
            Number of sampled input distributions for the discretized
            more-capable constraints.  Defaults to max(50, 10*|T|).
        niter : int or None
            Number of optimization restarts.
        rng : int or None
            Random seed for simplex samples.

        Returns
        -------
        float
            The more-capable II redundancy (in bits).
        """
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)

        channels, pi_t = _source_channels(d, sources, target)

        return _more_capable_ii(channels, pi_t, n_samples=n_samples, niter=niter, seed=rng)
