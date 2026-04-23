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
from scipy.optimize import minimize

from ...channelorder._utils import channels_from_joint
from ..pid import BaseBivariatePID

__all__ = ("PID_Deg",)


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
        Max cardinality of Q.  Defaults to Kolchinsky's bound.
    niter : int or None
        Number of random restarts per |Q| value.

    Returns
    -------
    float
        The degradation intersection information (in bits).
    """
    n_sources = len(channels)
    sizes = [ch.shape[1] for ch in channels]

    if bound is None:
        bound = sum(sizes) - n_sources + 1

    if niter is None:
        niter = 20

    best_mi = 0.0
    penalty_weight = 200.0

    for n_q in range(1, bound + 1):
        n_params = sum(s * n_q for s in sizes)

        def _neg_obj(params, _sizes=sizes, _nq=n_q):
            offset = 0
            k_qs = []
            for i in range(n_sources):
                lam = _params_to_stochastic(
                    params[offset : offset + _sizes[i] * _nq],
                    _sizes[i],
                    _nq,
                )
                offset += _sizes[i] * _nq
                k_qs.append(channels[i] @ lam)

            k_q = k_qs[0]
            penalty = sum(np.sum((k_qs[i] - k_q) ** 2) for i in range(1, n_sources))
            return -_mi_bits(pi_t, k_q) + penalty_weight * penalty

        for _ in range(niter):
            x0 = np.random.randn(n_params) * 0.5
            res = minimize(_neg_obj, x0, method="L-BFGS-B", options={"maxiter": 500})

            # Verify feasibility and record MI
            offset = 0
            k_qs = []
            for i in range(n_sources):
                lam = _params_to_stochastic(
                    res.x[offset : offset + sizes[i] * n_q],
                    sizes[i],
                    n_q,
                )
                offset += sizes[i] * n_q
                k_qs.append(channels[i] @ lam)

            penalty = sum(np.sum((k_qs[i] - k_qs[0]) ** 2) for i in range(1, n_sources))
            if penalty < 1e-6:
                mi = _mi_bits(pi_t, k_qs[0])
                if mi > best_mi:
                    best_mi = mi

    return best_mi


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
            Max cardinality of the auxiliary variable Q.
        niter : int or None
            Number of optimization restarts per |Q| value.

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
