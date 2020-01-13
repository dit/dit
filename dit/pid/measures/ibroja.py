# -*- coding: utf-8 -*-

"""
The I_BROJA unique measure, as proposed by the BROJA team.
"""

from ..pid import BaseUniquePID

from ...algorithms import BaseConvexOptimizer
from ...algorithms.distribution_optimizers import BaseDistOptimizer, BROJABivariateOptimizer
from ...multivariate import coinformation


class BROJAOptimizer(BaseDistOptimizer, BaseConvexOptimizer):
    """
    Optimizer for computing the max mutual information between
    inputs and outputs. In the bivariate case, this corresponds to
    maximizing the coinformation.
    """

    def __init__(self, dist, source, others, target, rv_mode=None):
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
        rv_mode : bool
            Unused, provided for compatibility with parent class.
        """
        dist = dist.coalesce((source,) + (sum(others, ()),) + (target,))
        constraints = [[0, 2], [1, 2]]
        super().__init__(dist, marginals=constraints, rv_mode=rv_mode)
        self._source = {0}
        self._others = {1}
        self._target = {2}

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
            opt_dist = broja.construct_dist()
            uniques[sources[0]] = coinformation(opt_dist, [[0], [2]], [1])
            uniques[sources[1]] = coinformation(opt_dist, [[1], [2]], [0])
        else:
            for source in sources:
                others = sum([i for i in sources if i != source], ())
                dm = d.coalesce([source, others, target])
                broja = BROJAOptimizer(dm, (0,), ((1,),), (2,))
                broja.optimize(niter=1, maxiter=maxiter)
                d_opt = broja.construct_dist()
                uniques[source] = coinformation(d_opt, [[0], [2]], [1])

        return uniques
