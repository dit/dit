"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from ...algorithms import BaseConvexOptimizer
from ...algorithms.distribution_optimizers import BaseDistOptimizer, BROJABivariateOptimizer
from ...algorithms.optimization import parallel_sweep
from ..pid import BaseUniquePID

__all__ = ("PID_BROJA",)


def _optimized_pmf(opt, cutoff=1e-6):
    """
    The cutoff + renormalized joint pmf ndarray of a solved distribution
    optimizer, matching what :meth:`construct_dist` produces but without
    round-tripping through the (xarray-backed) :class:`Distribution`. Used to
    read information measures off the optimum cheaply.
    """
    pmf = opt.construct_vector(opt._optima.copy())
    pmf[pmf < cutoff] = 0
    pmf /= pmf.sum()
    return pmf.reshape(opt._shape)


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
    """

    _name = "I_broja"

    @staticmethod
    def _measure(d, sources, target, maxiter=1000):
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

        Returns
        -------
        ibroja : dict
            The value of I_broja for each individual source.
        """
        uniques = {}
        if len(sources) == 2:
            broja = BROJABivariateOptimizer(d, list(sources), target)
            broja.optimize(niter=1, maxiter=maxiter)
            pmf = _optimized_pmf(broja)
            # I[source : target | other source], read directly off the joint
            # ndarray rather than through the (xarray-backed) Distribution.
            uniques[sources[0]] = float(broja._conditional_mutual_information({0}, {2}, {1})(pmf))
            uniques[sources[1]] = float(broja._conditional_mutual_information({1}, {2}, {0})(pmf))
        else:

            def _run(source, rng):
                others = sum((i for i in sources if i != source), ())
                dm = d.coalesce([source, others, target])
                broja = BROJAOptimizer(dm, (0,), ((1,),), (2,))
                broja.optimize(niter=1, maxiter=maxiter, rng=rng)
                pmf = _optimized_pmf(broja)
                # I[source : target | others]
                cmi = broja._conditional_mutual_information(broja._source, broja._target, broja._others)
                return float(cmi(pmf))

            for source, value in zip(sources, parallel_sweep(_run, sources), strict=True):
                uniques[source] = value

        return uniques
