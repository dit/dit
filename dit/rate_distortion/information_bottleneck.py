"""
Optimizers for computing information bottleneck points.
"""

from __future__ import division

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException

class InformationBottleneck(BaseAuxVarOptimizer):
    """
    Base optimizer for information bottleneck type calculations.
    """
    _shotgun = 10

    def __init__(self, dist, beta, alpha=1.0, rvs=None, crvs=None, bound=None, rv_mode=None):
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

        super(InformationBottleneck, self).__init__(dist=dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)
        if not 0.0 <= alpha <= 1.0:
            msg = "alpha must be in [0.0, 1.0]."
            raise ditException(msg)
        else:
            self._alpha = alpha
        if not 0.0 <= beta:
            msg = "beta must be non-negative."
            raise ditException(msg)
        else:
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

    def _objective(self):
        """
        Compute the appropriate objective.

        Returns
        -------
        objective : func
            The objective function.
        """

        def ib_objective(self, x):
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

        def gib_objective(self, x):
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

        def dib_objective(self, x):
            """
            Compute H[T | Z] - \\beta I[Y : T | Z]

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

        if np.isclose(self._alpha, 1.0):
            return ib_objective
        elif np.isclose(self._alpha, 0.0):
            return dib_objective
        else:
            return gib_objective

    @classmethod
    def functional(cls):
        """
        """
        def information_bottleneck(dist, beta):
            """
            """
            ib = cls(dist=dist, beta=beta)
            ib.optimize()
            pmf = ib.construct_joint(ib._optima)
            complexity = ib.complexity(pmf)
            relevance = ib.relevance(pmf)
            return complexity, relevance

        return information_bottleneck


class InformationBottleneckDivergence(InformationBottleneck):
    """
    """
    def __init__(self, divergence):
        """
        """
        super(InformationBottleneckDivergence, self).__init__()
        self.distortion = self._distortion(divergence)

    def _distortion(self, divergence):
        """
        """
        idx_xyz = (3,)
        idx_yzt = (0,)
        idx_xt = (1, 2)

        p_xzy_shape = (self._shape[0]*self._shape[2], self._shape[1])
        p_tzy_shape = (self._shape[3]*self._shape[2], self._shape[1])

        def distortion(pmf):
            """
            """
            p_xt = pmf.sum(axis=idx_xt)

            p_xyz = pmf.sum(axis=idx_xyz)
            p_yzt = pmf.sum(axis=idx_yzt)
            p_xzy = np.transpose(p_xyz, (0, 2, 1))
            p_tzy = np.transpose(p_yzt, (2, 1, 0))
            p_xzy = p_xzy.reshape(p_xzy_shape)
            p_tzy = p_tzy.reshape(p_tzy_shape)
            p_y_xz = p_xzy / p_xzy.sum(axis=1, keepdims=True)
            p_y_tz = p_tzy / p_tzy.sum(axis=1, keepdims=True)

            divs = np.asarray([divergence(a, b) for b in p_y_tz for a in p_y_xz]).reshape()
            dist = p_xt * divs
            return dist

        return distortion