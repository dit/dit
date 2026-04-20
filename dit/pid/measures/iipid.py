"""
The I-PID measure from Venkatesh, Gurushankar & Schamberg (2023).

Defines unique information via information deficiency:
    delta_I(M : X \\ Y) := sup_{P(T|M)} [ I(T; X) - I(T; Y) ]
where T -- M -- (X, Y) is a Markov chain.

Redundancy is then symmetrized as:
    RI_I(M : X; Y) = min{ I(M;X) - delta_I(M:X\\Y), I(M;Y) - delta_I(M:Y\\X) }

Reference:
    P. Venkatesh, K. Gurushankar, G. Schamberg,
    "Capturing and Interpreting Unique Information", arXiv:2302.11873, 2023.
"""

from ...algorithms.optimization import BaseAuxVarOptimizer, BaseConvexOptimizer
from ...multivariate import coinformation
from ..pid import BaseBivariatePID

__all__ = ("PID_IPID",)


class IPIDOptimizer(BaseConvexOptimizer, BaseAuxVarOptimizer):
    """
    Optimizer for the I-PID information deficiency.

    Finds the channel P(T|M) that maximizes I(T; source) - I(T; other),
    where the distribution is coalesced as [source, other, target].
    """

    construct_initial = BaseAuxVarOptimizer.construct_random_initial

    def __init__(self, dist, source, other, target, bound=None):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the information deficiency of.
        source : iterable
            The source variable (the one with unique information).
        other : iterable
            The other source variable.
        target : iterable
            The target variable (M).
        bound : int, None
            Bound on the cardinality of the auxiliary variable T.
            If None, defaults to |M|.
        """
        dist = dist.coalesce([source, other, target])
        super().__init__(dist, [[0], [1]], [2])

        target_idx = list(self._crvs)[0]
        t_size = self._shape[target_idx]
        bound = min(bound, t_size) if bound is not None else t_size

        self._construct_auxvars([(self._crvs, bound)])

        self._source_rv = {0}
        self._other_rv = {1}

    def _objective(self):
        """
        Minimize -(I(T; source) - I(T; other)) = I(T; other) - I(T; source).
        """
        mi_t_source = self._mutual_information(self._arvs, self._source_rv)
        mi_t_other = self._mutual_information(self._arvs, self._other_rv)

        def objective(self, x):
            pmf = self.construct_joint(x)
            return mi_t_other(pmf) - mi_t_source(pmf)

        return objective


def _information_deficiency(d, source, other, target, bound=None, niter=None):
    """
    Compute the information deficiency delta_I(M : source \\ other).

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
    bound : int, None
        Bound on |T|. Defaults to |M|.
    niter : int, None
        Number of optimization restarts.

    Returns
    -------
    delta : float
        The information deficiency (non-negative).
    """
    opt = IPIDOptimizer(d, source, other, target, bound=bound)
    opt.optimize(niter=niter)
    delta = -opt.objective(opt._optima)
    return max(delta, 0.0)


class PID_IPID(BaseBivariatePID):
    """
    The I-PID from Venkatesh, Gurushankar & Schamberg (2023).

    Unique information is defined via information deficiency:
        delta_I(M : X \\ Y) = sup_{P(T|M)} [ I(T;X) - I(T;Y) ]
    and redundancy is symmetrized:
        RI_I = min{ I(M;X) - delta_I(M:X\\Y), I(M;Y) - delta_I(M:Y\\X) }
    """

    _name = "I_IPID"

    @staticmethod
    def _measure(d, sources, target, bound=None, niter=None):
        """
        Compute the I-PID redundancy for a pair of sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_IPID for.
        sources : iterable of iterables
            The source variables (exactly two).
        target : iterable
            The target variable.
        bound : int, None
            Bound on the cardinality of the auxiliary variable T.
        niter : int, None
            Number of optimization restarts.

        Returns
        -------
        ri : float
            The I-PID redundancy.
        """
        source_a, source_b = sources

        delta_ab = _information_deficiency(d, source_a, source_b, target,
                                           bound=bound, niter=niter)
        delta_ba = _information_deficiency(d, source_b, source_a, target,
                                           bound=bound, niter=niter)

        mi_a = coinformation(d, [source_a, target])
        mi_b = coinformation(d, [source_b, target])

        return min(mi_a - delta_ab, mi_b - delta_ba)
