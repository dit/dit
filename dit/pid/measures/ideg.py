"""
The degradation intersection information I_d^∩ from Kolchinsky (2022).

Defines redundancy as the maximum information a channel K_Q can carry
about T while remaining a Blackwell degradation of every source channel:

    I_d^∩(Y1, ..., Yn → T) = max_{Q : Q ≤_d Y_i ∀i} I(Q; T)

The feasible set is a polytope (linear constraints on the degradation
channels Λ_i with K_Q = K^(i) @ Λ_i), and I(Q;T) is convex in K_Q,
so the maximum is attained at a vertex.

References
----------
.. [1] A. Kolchinsky, "A Novel Approach to the Partial Information
       Decomposition", Entropy 24, 403, 2022.
.. [2] A. F. C. Gomes and M. A. T. Figueiredo, "Orders between Channels
       and Implications for Partial Information Decomposition",
       Entropy 25, 975, 2023.
"""

import numpy as np
from scipy.optimize import linprog

from ...channelorder._utils import channels_from_joint
from ..pid import BaseBivariatePID

__all__ = ("PID_Deg",)


def _mi_bits(pi_t, ch):
    """I(T; Q) in bits for channel ch = P(Q|T) given input dist pi_t."""
    eps = 1e-300
    p_q = pi_t @ ch
    mi = 0.0
    for t in range(len(pi_t)):
        for q in range(ch.shape[1]):
            pj = pi_t[t] * ch[t, q]
            if pj > eps:
                mi += pj * np.log2(ch[t, q] / (p_q[q] + eps))
    return mi


def _mi_grad(pi_t, ch, floor=1e-12):
    """Gradient of I(T; Q) w.r.t. the channel, with logs floored so zero entries stay finite."""
    p_q = pi_t @ ch
    return pi_t[:, None] * np.log2(np.maximum(ch, floor) / np.maximum(p_q, floor)[None, :])


def _degradation_channel(channels, pi_t, bound=None, niter=None, steps=50, seed=0):
    """
    Maximize I(Q; T) over common degradations K_Q = K^(i) @ Λ_i of every source.

    The feasible (Λ_1, ..., Λ_n) form a polytope and I(Q; T) is convex in K_Q, so
    the maximum is at a vertex. Each start is a vertex from a linear program with a
    random cost, followed by successive linearization: move to the vertex maximizing
    the gradient's linear form while that improves I(Q; T). Every candidate is
    exactly feasible.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).
    bound : int or None
        Cardinality of Q.  Defaults to Kolchinsky's bound.
    niter : int or None
        Number of random vertices to start from.
    steps : int
        Maximum linearization steps per start.
    seed : int or None
        Random seed for the vertex costs.

    Returns
    -------
    mi : float
        The degradation intersection information (in bits).
    k_q : ndarray or None
        An optimal channel K_Q, or None if the optimum is the constant channel.
    """
    n_t = channels[0].shape[0]
    sizes = [ch.shape[1] for ch in channels]
    n_q = bound if bound is not None else sum(sizes) - len(channels) + 1
    niter = 100 if niter is None else niter
    offsets = np.cumsum([0] + [s * n_q for s in sizes])
    n_z = offsets[-1]

    def kq_map(i):
        """Linear map from the stacked Λ's to vec(K^(i) @ Λ_i)."""
        M = np.zeros((n_t * n_q, n_z))
        block = np.kron(channels[i], np.eye(n_q))
        M[:, offsets[i] : offsets[i + 1]] = block
        return M

    rows = [np.kron(np.eye(s), np.ones(n_q)) for s in sizes]
    A_rows = np.zeros((sum(sizes), n_z))
    r = 0
    for i, block in enumerate(rows):
        A_rows[r : r + sizes[i], offsets[i] : offsets[i + 1]] = block
        r += sizes[i]
    M0 = kq_map(0)
    A = np.vstack([A_rows] + [M0 - kq_map(i) for i in range(1, len(channels))])
    b = np.r_[np.ones(sum(sizes)), np.zeros(A.shape[0] - sum(sizes))]

    def channel(z):
        return np.clip((M0 @ z).reshape(n_t, n_q), 0, None)

    rng = np.random.default_rng(seed)
    best_mi, best_k = 0.0, None
    for _ in range(niter):
        res = linprog(rng.standard_normal(n_z), A_eq=A, b_eq=b, bounds=(0, None), method="highs-ds")
        if res.x is None:
            continue
        z = res.x
        val = _mi_bits(pi_t, channel(z))
        for _ in range(steps):
            g = M0.T @ _mi_grad(pi_t, channel(z)).ravel()
            res = linprog(-g, A_eq=A, b_eq=b, bounds=(0, None), method="highs-ds")
            if res.x is None or _mi_bits(pi_t, channel(res.x)) <= val + 1e-12:
                break
            z, val = res.x, _mi_bits(pi_t, channel(res.x))
        if val > best_mi:
            best_mi, best_k = val, channel(z)
    return best_mi, best_k


def _degradation_ii(channels, pi_t, bound=None, niter=None):
    """
    Compute I_d^∩ by maximizing I(Q; T) subject to Q ≤_d Y_i for all i.

    Parameters
    ----------
    channels : list of ndarray
        Source channel matrices [K^(1), K^(2), ...], each shape (|T|, |Y_i|).
    pi_t : ndarray
        Target marginal P(T).
    bound : int or None
        Cardinality of Q.  Defaults to Kolchinsky's bound.
    niter : int or None
        Number of random vertices to start from.

    Returns
    -------
    float
        The degradation intersection information (in bits).
    """
    return _degradation_channel(channels, pi_t, bound=bound, niter=niter)[0]


class PID_Deg(BaseBivariatePID):
    """
    Degradation intersection information I_d^∩ from Kolchinsky (2022).

    Redundancy is the maximum I(Q; T) such that Q is a Blackwell
    degradation of every source channel.  Satisfies the Williams-Beer
    axioms and the independent identity property (IIP).

    References
    ----------
    .. [1] A. Kolchinsky, "A Novel Approach to the Partial Information
           Decomposition", Entropy 24, 403, 2022.
    """

    _name = "I_d∩"

    @staticmethod
    def _measure(d, sources, target, bound=None, niter=None):
        """
        Compute the degradation II redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The joint distribution.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.
        bound : int or None
            Cardinality of the auxiliary variable Q.
        niter : int or None
            Number of random vertices to start the search from.

        Returns
        -------
        float
            The degradation II redundancy (in bits).
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

        return _degradation_ii(
            [kappa_a, kappa_b],
            pi_t,
            bound=bound,
            niter=niter,
        )
