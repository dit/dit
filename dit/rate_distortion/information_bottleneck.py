"""
"""

from __future__ import division

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException

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
