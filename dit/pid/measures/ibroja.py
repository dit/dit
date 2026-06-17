"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from ...algorithms import BaseConvexOptimizer
from ...algorithms.broja_method import broja_solve_bivariate
from ...algorithms.broja_util import optimized_pmf
from ...algorithms.distribution_optimizers import BaseDistOptimizer
from ...algorithms.optimization import parallel_sweep
from ..pid import BaseUniquePID

__all__ = ("PID_BROJA",)


class BROJAOptimizer(BaseDistOptimizer, BaseConvexOptimizer):
    """
    Optimizer for computing the max mutual information between
    inputs and outputs. In the bivariate case, this corresponds to
    maximizing the coinformation.
    """

    def __init__(self, dist, source, others, target):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to base the optimization on.
        source : iterable
            Variable to treat as the source.
        others : iterable of iterables
            The other source variables.
        target : iterable
            The target variable.
        """
        dist = dist.coalesce((source,) + (sum(others, ()),) + (target,))
        constraints = [[0, 2], [1, 2]]
        super().__init__(dist, marginals=constraints)
        self._source = {0}
        self._others = {1}
        self._target = {2}

    def _objective_gradient(self):
        """Gradient of the ``I[source : target | others]`` objective w.r.t. the joint."""
        return self._conditional_mutual_information_grad(self._source, self._target, self._others)

    def _objective(self):
        """
        Minimize I(source:target|others).

        Parameters
        ----------
        x : np.ndarray
            Optimization vector.

        Returns
        -------
        obj : func
            The objective.
        """
        cmi = self._conditional_mutual_information(self._source, self._target, self._others)

        def objective(self, x):
            """
            Compute I[source : target | others]

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            pmf = self.construct_joint(x)
            return cmi(pmf)

        return objective


class PID_BROJA(BaseUniquePID):
    """
    The BROJA partial information decomposition.

    Notes
    -----
    This partial information decomposition, at least in the bivariate source
    case, was independently suggested by Griffith.

    For two sources, ``method`` selects the bivariate solver:

    - ``'scipy'`` — SLSQP on the marginal-matching polytope (default for small alphabets)
    - ``'admui'`` — alternating divergence minimization (:cite:`banerjee2017computing`)
    - ``'cone'`` — exponential cone program via ECOS (:cite:`makkeh2018broja`)
    - ``'auto'`` — size-based selection with scipy/cone fallbacks
    """

    _name = "I_broja"

    @staticmethod
    def _measure(d, sources, target, maxiter=1000, method="auto", rng=None, **ecos_kwargs):
        """
        This computes unique information as min{I(source : target | other_sources)}
        over the space of distributions which matches source-target marginals.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_broja for.
        sources : iterable of iterables
            The target variables.
        target : iterable
            The target variable.
        maxiter : int
            Maximum iterations for scipy/admui solvers.
        method : str
            Bivariate solver: ``'scipy'``, ``'admui'``, ``'cone'``, or ``'auto'``.
        rng : np.random.Generator, optional
            RNG for scipy restarts.
        **ecos_kwargs
            Passed to ECOS when ``method='cone'``.

        Returns
        -------
        ibroja : dict
            The value of I_broja for each individual source.
        """
        if len(sources) == 2:
            uniques, _meta = broja_solve_bivariate(
                d,
                sources,
                target,
                maxiter=maxiter,
                method=method,
                rng=rng,
                **ecos_kwargs,
            )
            return uniques

        uniques = {}

        def _run(source, rng):
            others = sum((i for i in sources if i != source), ())
            dm = d.coalesce([source, others, target])
            broja = BROJAOptimizer(dm, (0,), ((1,),), (2,))
            broja.optimize(niter=1, maxiter=maxiter, rng=rng)
            pmf = optimized_pmf(broja)
            cmi = broja._conditional_mutual_information(broja._source, broja._target, broja._others)
            return float(cmi(pmf))

        for source, value in zip(sources, parallel_sweep(_run, sources), strict=True):
            uniques[source] = value

        return uniques
