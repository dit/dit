"""
"""

from __future__ import division

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException
from ..multivariate import entropy, total_correlation

class BaseInformationBottleneck(BaseAuxVarOptimizer):
    """
    Base optimizer for information bottleneck type calculations.
    """
    _shotgun = 10

    def __init__(self, dist, beta=1.0, rvs=None, crvs=None, bound=None, rv_mode=None):
        """
        Initialize the bottleneck.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        beta : float
            The beta value used in the objective function.
        rvs : iter of iters, None
            The random variables to compute the bottleneck of.
        crvs : iter, None
            The random variables to condition on.
        bound : int, None
            The bound on the size of the statistic. If None, use the size of X.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        if rvs is None:
            rvs = dist.rvs

        if len(rvs) != 2:
            msg = "The information bottleneck is only defined for two variables."
            raise ditException(msg)

        super(BaseInformationBottleneck, self).__init__(dist=dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)
        self._beta = beta

        self._x = {0}
        self._y = {1}
        self._z = {2}
        self._t = {3}

        tbound = self._shape[0] * self._shape[2]
        # tbound = int(np.ceil(perplexity(dist, rvs[0])))
        bound = min([bound, tbound]) if bound is not None else tbound

        self._construct_auxvars([(self._x | self._z, tbound)])

        self.complexity = self._conditional_mutual_information(self._x, self._t, self._z)
        self.entropy = self._entropy(self._t, self._z)
        self.relevance = self._conditional_mutual_information(self._y, self._t, self._z)
        self.other = self._entropy(self._t, self._x | self._z)
        self.error = self._conditional_mutual_information(self._x, self._y, self._t | self._z)

        self._default_hops *= 2

    @classmethod
    def functional(cls):
        """
        """
        pass


class InformationBottleneck(BaseInformationBottleneck):
    """
    The information bottleneck:

        DIB_\\beta[X : Y | Z] = \min_{T - XZ - Y} H[T | Z] - \\beta I[Y : T | Z]
    """
    def _objective(self):
        """
        Compute the IB objective.

        Returns
        -------
        objective : func
            The objective function.
        """

        def objective(self, x):
            """
            Compute I[X : T | Z] - \\beta I[Y : T | Z]

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
            obj = self.complexity(pmf) - self._beta * self.relevance(pmf)
            return obj

        return objective


class DeterministicInformationBottleneck(BaseInformationBottleneck):
    """
    The deterministic information bottleneck:

        DIB_\\beta[X : Y | Z] = \min_{T - XZ - Y} H[T | Z] - \\beta I[Y : T | Z]
    """
    def _objective(self):
        """
        Compute the DIB objective.

        Returns
        -------
        objective : func
            The objective function.
        """

        def objective(self, x):
            """
            Compute I[X : T | Z] - \\beta I[Y : T | Z]

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
            obj = self.entropy(pmf) - self._beta * self.relevance(pmf)
            return obj

        return objective


class GeneralizedInformationBottleneck(BaseInformationBottleneck):
    """
    The deterministic information bottleneck:

        DIB_\\beta[X : Y | Z] = \min_{T - XZ - Y} H[T | Z] - \\alpha H[T | X, Z]- \\beta I[Y : T | Z]
    """

    def __init__(self, dist, alpha=1.0, beta=1.0, rvs=None, crvs=None, bound=None, rv_mode=None):
        """
        Initialize the bottleneck.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        alpha : float, 0 <= `alpha` <= 1
            The alpha value used in the objective function.
        beta : float
            The beta value used in the objective function.
        rvs : iter of iters, None
            The random variables to compute the bottleneck of.
        crvs : iter, None
            The random variables to condition on.
        bound : int, None
            The bound on the size of the statistic. If None, use the size of X.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        super(GeneralizedInformationBottleneck, self).__init__(dist, beta=beta, rvs=rvs, crvs=crvs, bound=bound, rv_mode=rv_mode)
        self._alpha = alpha

    def _objective(self):
        """
        Compute the GIB objective.

        Returns
        -------
        objective : func
            The objective function.
        """

        def objective(self, x):
            """
            Compute H[T | Z] - \\alpha H[T | X, Z]- \\beta I[Y : T | Z]

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
            obj = self.entropy(pmf) - self._alpha * self.other(pmf) - self._beta * self.relevance(pmf)
            return obj

        return objective


class IBCurve(object):
    """
    """

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None, beta_min=0.0, beta_max=15.0, beta_num=101, alpha=1.0, bound=None):
        """
        """
        self.dist = dist.copy()

        self._x, self._y = rvs if rvs is not None else ([0], [1])
        self._z = crvs if crvs is not None else []
        self._aux = [dist.outcome_length()]
        self._rv_mode = rv_mode

        self._true_complexity = entropy(dist, self._x)  # TODO: replace with MSS
        self._true_relevance = total_correlation(dist, [self._x, self._y])

        beta_max = self.find_max_beta() if beta_max is None else beta_max
        self.betas = np.linspace(beta_min, beta_max, beta_num)

        self._args = {'dist': self.dist,
                      'rvs': [self._x, self._y],
                      'crvs': self._z,
                      'bound': bound,
                      'rv_mode': self._rv_mode,
                      }

        if np.isclose(alpha, 1.0):
            self._bottleneck = InformationBottleneck
        elif np.isclose(alpha, 0.0):
            self._bottleneck = DeterministicInformationBottleneck
        else:
            self._bottleneck = GeneralizedInformationBottleneck
            self._args['alpha'] = alpha

        self.compute()

    def compute(self):
        """
        """
        self.complexities = []
        self.entropies = []
        self.relevances = []
        self.errors = []

        x0 = None

        for beta in self.betas:
            ib = self._bottleneck(beta=beta, **self._args)
            ib.optimize(x0=x0)
            x0 = ib._optima.copy()
            pmf = ib.construct_joint(ib._optima)
            self.complexities.append(ib.complexity(pmf))
            self.entropies.append(ib.entropy(pmf))
            self.relevances.append(ib.relevance(pmf))
            self.errors.append(ib.error(pmf))

    def find_max_beta(self):
        """
        """
        beta_max = 2/3
        relevance = 0

        while not np.isclose(relevance, self._true_relevance, atol=1e-3, rtol=1e-3):
            beta_max = int(np.ceil(1.5*beta_max))
            ib = self._bottleneck(self.dist,
                                  beta=beta_max,
                                  rvs=[self._x, self._y],
                                  crvs=self._z,
                                  rv_mode=self._rv_mode
                                 )
            ib.optimize()
            d = ib.construct_distribution()
            relevance = total_correlation(d, [self._y, self._aux], self._z)

        return beta_max

    def find_kinks(self):
        """
        """
        pass

    def plot(self, downsample=5):
        """
        """
        from .plotting import IBPlotter
        plotter = IBPlotter(self)
        return plotter.plot(downsample)

    def __add__(self, other):
        """
        """
        from .plotting import IBPlotter
        if isinstance(other, IBCurve):
            plotter = IBPlotter(self, other)
            return plotter
        else:
            return NotImplemented
