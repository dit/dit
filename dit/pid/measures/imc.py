"""
The more-capable intersection information I_mc^∩ from Gomes & Figueiredo (2023).

Defines redundancy as:

    I_mc^∩(Y1, ..., Yn → T) = max_{Q : Q ≤_mc Y_i ∀i} I(Q; T)

where ≤_mc is the "more capable" channel preorder: K_Q ≤_mc K^(i) iff
I(Q; T) ≤ I(Y_i; T) for *every* input distribution P(T).

The more-capable constraint set is convex but involves an uncountable
family of inequalities.  Following the paper (Section 5.2, eq. 13), we
discretize the input simplex with S sampled distributions and solve the
resulting finite-constraint problem, which yields an upper bound that
tightens as S grows.

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

from ...channelorder._utils import channels_from_joint
from ..pid import BaseBivariatePID

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
    eps = 1e-300
    p_out = pi_t @ ch
    mi = 0.0
    for t in range(len(pi_t)):
        for q in range(ch.shape[1]):
            pj = pi_t[t] * ch[t, q]
            if pj > eps:
                mi += pj * np.log2(ch[t, q] / (p_out[q] + eps))
    return mi


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
        Upper bound on I_mc^∩ (in bits).  Tightens as n_samples grows.
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

    # Pre-compute min_i I(Y_i; T) for each sampled distribution
    mi_bounds = np.zeros(len(sampled_pi))
    for k, pi_k in enumerate(sampled_pi):
        mi_bounds[k] = min(_mi_bits(pi_k, ch) for ch in channels)

    atol = 1e-6
    best_mi = 0.0

    def _check_and_update(k_q):
        nonlocal best_mi
        for k in range(len(sampled_pi)):
            if _mi_bits(sampled_pi[k], k_q) - mi_bounds[k] > atol:
                return
        mi = _mi_bits(pi_t, k_q)
        if mi > best_mi:
            best_mi = mi

    # Phase 1: try each source channel directly (always ≤_mc itself)
    for ch in channels:
        _check_and_update(ch)

    # Phase 2: SLSQP with inequality constraints from multiple starts
    n_params = n_t * n_q

    def _objective(params):
        k_q = _params_to_stochastic(params, n_t, n_q)
        return -_mi_bits(pi_t, k_q)

    constraint_list = []
    for k_idx in range(len(sampled_pi)):

        def _con(params, _k=k_idx):
            k_q = _params_to_stochastic(params, n_t, n_q)
            return mi_bounds[_k] - _mi_bits(sampled_pi[_k], k_q)

        constraint_list.append({"type": "ineq", "fun": _con})

    # Initial points: near each source channel, plus random
    init_points = []
    for ch in channels:
        if ch.shape[1] == n_q:
            init_points.append(np.log(ch + 1e-10).ravel())
        elif ch.shape[1] < n_q:
            padded = np.full((n_t, n_q), 1e-10)
            padded[:, : ch.shape[1]] = ch
            init_points.append(np.log(padded + 1e-10).ravel())
    for _ in range(max(1, niter - len(init_points))):
        init_points.append(rng.standard_normal(n_params) * 0.5)

    for x0 in init_points:
        try:
            res = minimize(
                _objective, x0, method="SLSQP", constraints=constraint_list, options={"maxiter": 300, "ftol": 1e-12}
            )
            k_q = _params_to_stochastic(res.x, n_t, n_q)
            _check_and_update(k_q)
        except Exception:
            continue

    return best_mi


class PID_MC(BaseBivariatePID):
    """
    More-capable intersection information I_mc^∩ from Gomes & Figueiredo.

    Redundancy is the maximum I(Q; T) such that K_Q is "less capable"
    than every source channel for all input distributions.

    Because the constraint set is discretized, the computed value is an
    upper bound that tightens with more samples.

    References
    ----------
    .. [1] A. F. C. Gomes and M. A. T. Figueiredo, "Orders between
           Channels and Implications for Partial Information
           Decomposition", Entropy 25, 975, 2023.
    """

    _name = "I_mc∩"

    @staticmethod
    def _measure(d, sources, target, n_samples=None, niter=None, seed=None):
        """
        Compute the more-capable II redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.
        n_samples : int or None
            Number of sampled input distributions for the discretized
            more-capable constraints.  Defaults to max(50, 10*|T|).
        niter : int or None
            Number of optimization restarts.
        seed : int or None
            Random seed for simplex samples.

        Returns
        -------
        float
            The more-capable II redundancy (upper bound, in bits).
        """
        source_a, source_b = sources

        d_coal = d.coalesce([list(source_a), list(source_b), list(target)])
        src_a, src_b, tgt = d_coal.dims

        kappa_a, kappa_b, pi_t = channels_from_joint(
            d_coal,
            [tgt],
            [src_a],
            [src_b],
        )

        return _more_capable_ii(
            [kappa_a, kappa_b],
            pi_t,
            n_samples=n_samples,
            niter=niter,
            seed=seed,
        )
