"""
The delta-lambda PID: a Lagrangian generalization interpolating between the
delta-PID (Banerjee et al. 2018) and the BROJA/tilde-PID (Bertschinger et al. 2014).

Defines the generalized deficiency as:
    delta_lambda(M : X \\ Y) := inf_{P(X'|MY)} E_M[D_KL(P(X|M) || P(X'|M))]
                                + lambda * I(M; X' | Y)

As lambda -> infinity, this recovers the delta-PID (enforces Markov chain
M--Y--X', measures KL departure from copy).
As lambda -> 0, this recovers the BROJA/tilde-PID (enforces copy, measures
departure from Markov chain).

Redundancy is min-symmetrized:
    RI_lambda(M : X; Y) = min{ I(M;X) - delta_lambda(M:X\\Y),
                                I(M;Y) - delta_lambda(M:Y\\X) }

References
----------
.. [1] P. Venkatesh, K. Gurushankar, G. Schamberg,
       "Capturing and Interpreting Unique Information", arXiv:2302.11873, 2023.
       Section III-G, Equation (26).
"""

import numpy as np

from ...algorithms.optimization import BaseAuxVarOptimizer, BaseConvexOptimizer, parallel_sweep
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_DeltaLambda",)


class DeltaLambdaOptimizer(BaseConvexOptimizer, BaseAuxVarOptimizer):
    """
    Optimizer for the delta-lambda generalized deficiency.

    Minimizes E_M[D_KL(P(X|M) || P(X'|M))] + lambda * I(M; X' | Y)
    over the channel P(X'|M,Y), where:
    - source (X) is variable 0
    - other (Y) is variable 1
    - target (M) is variable 2
    - auxiliary (X') is variable 3, conditioned on (M, Y) = {1, 2}
    """

    construct_initial = BaseAuxVarOptimizer.construct_random_initial

    def __init__(self, dist, source, other, target, lam=1.0, bound=None):
        """
        Parameters
        ----------
        dist : Distribution
            The joint distribution.
        source : iterable
            The source variable (X).
        other : iterable
            The other source variable (Y).
        target : iterable
            The target variable (M).
        lam : float
            Lagrange multiplier. Large values enforce the Markov chain
            (approaching delta-PID), small values enforce the copy
            (approaching BROJA).
        bound : int, None
            Bound on |X'|. Defaults to |X|.
        """
        dist = dist.coalesce([source, other, target])
        super().__init__(dist, [[0], [1]], [2])

        source_size = self._shape[0]
        bound = min(bound, source_size) if bound is not None else source_size

        # X' conditioned on (other=1, target=2) with alphabet size = |source|
        self._construct_auxvars([({1, 2}, bound)])

        self._lam = lam

        self._source_rv = {0}
        self._other_rv = {1}
        self._target_rv = {2}
        self._aux_rv = self._arvs

    def _objective_gradient(self):
        """
        Analytic gradient of the delta-lambda objective w.r.t. the joint.

        The Lagrangian is ``kl + lam * I(X':M|Y)`` where
        ``kl = sum_{a,m} A[a,m] (log2 A[a,m] - log2 B[m,a])`` with
        ``A = p(X, M)`` (sum over Y, X') and ``B = p(X', M)`` (sum over X, Y).
        Differentiating term by term:

        * ``d kl / d A[a,m] = log2 A[a,m] - log2 B[m,a] + 1/ln2``
        * ``d kl / d B[m,a] = -A[a,m] / (B[m,a] * ln2)``

        and ``A`` / ``B`` are linear marginals of the joint, so each cell's
        gradient is the appropriate broadcast of those two terms. The CMI term
        reuses the standard conditional-MI gradient builder.
        """
        cmi_g = self._conditional_mutual_information_grad(self._aux_rv, self._target_rv, self._other_rv)
        lam = self._lam
        ln2 = np.log(2)

        def grad(pmf):
            # A = p(X, M): sum over Y(1), X'(3) -> (|X|, |M|)
            A = pmf.sum(axis=(1, 3))
            # B = p(X', M): sum over X(0), Y(1) -> (|M|, |X'|)
            B = pmf.sum(axis=(0, 1))

            with np.errstate(divide="ignore"):
                log_A = np.log2(np.maximum(A, 1e-300))  # (|X|, |M|)
                log_B_aligned = np.log2(np.maximum(B, 1e-300)).T  # (|X'|=|X|, |M|)

            # contribution flowing through A[x, m], broadcast over Y and X'.
            a_term = log_A - log_B_aligned + 1.0 / ln2  # (|X|, |M|)
            g_a = a_term.reshape(A.shape[0], 1, A.shape[1], 1)

            # contribution flowing through B[m, x'], broadcast over X and Y.
            b_term = -A.T / (np.maximum(B, 1e-300) * ln2)  # (|M|, |X'|)
            g_b = b_term.reshape(1, 1, B.shape[0], B.shape[1])

            kl_grad = np.broadcast_to(g_a, pmf.shape) + np.broadcast_to(g_b, pmf.shape)
            return np.ascontiguousarray(kl_grad) + lam * cmi_g(pmf)

        return grad

    def _objective(self):
        """
        Minimize E_M[D_KL(P(X|M) || P(X'|M))] + lambda * I(M; X' | Y).
        """
        cmi_func = self._conditional_mutual_information(self._aux_rv, self._target_rv, self._other_rv)
        lam = self._lam
        source_axis = 0
        other_axis = 1
        aux_axis = 3

        def objective(self, x):
            pmf = self.construct_joint(x)

            # P(X, M): marginalize out Y and X'
            p_xm = pmf.sum(axis=(other_axis, aux_axis))
            # P(X', M): marginalize out X and Y
            p_xprime_m = pmf.sum(axis=(source_axis, other_axis))

            # E_M[D_KL(P(X|M) || P(X'|M))]
            # = sum_{a,m} P(X=a, M=m) * log2(P(X=a, M=m) / P(X'=a, M=m))
            # p_xprime_m has shape (|M|, |X'|), transpose to (|X'|, |M|) = (|X|, |M|)
            p_xprime_aligned = p_xprime_m.T

            mask = p_xm > 0
            log_ratio = np.where(
                mask,
                np.log2(p_xm + 1e-300) - np.log2(p_xprime_aligned + 1e-300),
                0.0,
            )
            kl_term = np.sum(p_xm * log_ratio)

            cmi_term = cmi_func(pmf)

            return kl_term + lam * cmi_term

        return objective


def _delta_lambda(d, source, other, target, lam=1.0, bound=None, niter=None, rng=None):
    """
    Compute the generalized deficiency delta_lambda(M : source \\ other).

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    source : iterable
        The source with unique information.
    other : iterable
        The other source.
    target : iterable
        The target variable.
    lam : float
        Lagrange multiplier.
    bound : int, None
        Bound on |X'|. Defaults to |source|.
    niter : int, None
        Number of optimization restarts.

    Returns
    -------
    delta : float
        The generalized deficiency (non-negative).
    """
    opt = DeltaLambdaOptimizer(d, source, other, target, lam=lam, bound=bound)
    opt.optimize(niter=niter, rng=rng)
    return max(opt.objective(opt._optima), 0.0)


class PID_DeltaLambda(BaseBivariatePID):
    r"""
    The delta-lambda PID from Venkatesh, Gurushankar & Schamberg (2023).

    A Lagrangian generalization that interpolates between the delta-PID
    and the BROJA PID via a parameter lambda:

        delta_lambda(M : X \ Y) = inf_{P(X'|MY)} E_M[D_KL(P(X|M) || P(X'|M))]
                                  + lambda * I(M; X' | Y)

    Redundancy is symmetrized:
        RI = min{ I(M;X) - delta_lambda(M:X\Y), I(M;Y) - delta_lambda(M:Y\X) }

    Parameters (pass via keyword arguments to constructor)
    ----------
    lam : float
        Lagrange multiplier (default 1.0). Large values approach the
        delta-PID; small values approach the BROJA PID.
    bound : int, None
        Bound on auxiliary variable cardinality.
    niter : int, None
        Number of optimization restarts.
    """

    _name = "I_\u03b4\u03bb"

    @staticmethod
    def _measure(d, sources, target, lam=1.0, bound=None, niter=None):
        """
        Compute the delta-lambda PID redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute the PID for.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.
        lam : float
            Lagrange multiplier.
        bound : int, None
            Bound on auxiliary variable cardinality.
        niter : int, None
            Number of optimization restarts.

        Returns
        -------
        ri : float
            The delta-lambda PID redundancy.
        """
        source_a, source_b = sources

        def _run(args, rng):
            src, oth = args
            return _delta_lambda(d, src, oth, target, lam=lam, bound=bound, niter=niter, rng=rng)

        delta_ab, delta_ba = parallel_sweep(_run, [(source_a, source_b), (source_b, source_a)])

        mi_a = coinformation(d, [source_a, target])
        mi_b = coinformation(d, [source_b, target])

        return min(mi_a - delta_ab, mi_b - delta_ba)
