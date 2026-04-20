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

from ...algorithms.optimization import BaseAuxVarOptimizer, BaseConvexOptimizer
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

    def _objective(self):
        """
        Minimize E_M[D_KL(P(X|M) || P(X'|M))] + lambda * I(M; X' | Y).
        """
        cmi_func = self._conditional_mutual_information(
            self._aux_rv, self._target_rv, self._other_rv
        )
        lam = self._lam
        source_axis = 0
        other_axis = 1
        target_axis = 2
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


def _delta_lambda(d, source, other, target, lam=1.0, bound=None, niter=None):
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
    opt.optimize(niter=niter)
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

        delta_ab = _delta_lambda(d, source_a, source_b, target,
                                 lam=lam, bound=bound, niter=niter)
        delta_ba = _delta_lambda(d, source_b, source_a, target,
                                 lam=lam, bound=bound, niter=niter)

        mi_a = coinformation(d, [source_a, target])
        mi_b = coinformation(d, [source_b, target])

        return min(mi_a - delta_ab, mi_b - delta_ba)
