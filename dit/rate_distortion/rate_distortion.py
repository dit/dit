"""
Optimizers for computing rate-distortion points.
"""

from abc import abstractmethod
from collections import namedtuple

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException
from ..utils import flatten


RateDistortionResult = namedtuple('RateDistortionResult', ['rate', 'distortion'])


class BaseRateDistortion(BaseAuxVarOptimizer):
    """
    Base optimizer for rate distortion type calculations.
    """
    _shotgun = 10

    def __init__(self, dist, beta, alpha=1.0, rv=None, crvs=None, bound=None, rv_mode=None):
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
        if rv is None:
            rv = list(flatten(dist.rvs))

        try:  # pragma: no cover
            iter(rv[0])
            msg = "Rate-Distortion is only defined for one variable."
            raise ditException(msg)
        except TypeError:
            pass

        super(BaseRateDistortion, self).__init__(dist=dist, rvs=[rv], crvs=crvs, rv_mode=rv_mode)
        self._alpha = alpha
        self._beta = beta

        self._x = {0}
        self._z = {1}
        self._t = {2}

        # tbound = self._shape[0] * self._shape[2]
        tbound = self._shape[0]
        bound = min([bound, tbound]) if bound is not None else tbound

        self._construct_auxvars([(self._x | self._z, tbound)])

        self.rate = self._conditional_mutual_information(self._x, self._t, self._z)
        self.distortion = self._distortion()
        self.entropy = self._entropy(self._t, self._z)
        self.other = self._entropy(self._t, self._x | self._z)

        self._default_hops *= 2

    def _objective(self):
        """
        Produce the rate-distortion Lagrangian.
        """
        def rd_objective(self, x):
            """
            The rate-distortion objective.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The objective value.
            """
            pmf = self.construct_joint(x)
            obj = self.rate(pmf) + self._beta * self.distortion(pmf)
            return obj

        def grd_objective(self, x):
            """
            The generalized rate-distortion objective.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The objective value.
            """
            pmf = self.construct_joint(x)
            obj = self.entropy(pmf) - self._alpha * self.other(pmf) + self._beta * self.distortion(pmf)
            return obj

        def drd_objective(self, x):
            """
            The deterministic rate-distortion objective.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The objective value.
            """
            pmf = self.construct_joint(x)
            obj = self.entropy(pmf) + self._beta * self.distortion(pmf)
            return obj

        if np.isclose(self._alpha, 1.0):
            return rd_objective
        elif np.isclose(self._alpha, 0.0):
            return drd_objective
        else:
            return grd_objective

    @abstractmethod
    def _distortion(self):
        """
        Return a distortion measure.

        Returns
        -------
        distortion : func
            A distortion measure.
        """
        pass

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """
        def rate_distortion(dist, beta=0.0, rv=None, crvs=None, bound=None, rv_mode=None):
            """
            Compute a rate-distortion point.

            Parameters
            ----------
            dist : Distribution
                The distribution of interest.
            beta : float
                The beta value to optimize with.
            rv : iterable, None
                The indices to consider. If None, use all.
            crvs : iterable, None
                The indices to condition on. If None, use none.
            bound : int, None
                Bound the size of the compressed variable.
            rv_mode : str, None
                Specifies how to interpret `rvs` and `crvs`. Valid options are:
                {'indices', 'names'}. If equal to 'indices', then the elements of
                `crvs` and `rvs` are interpreted as random variable indices. If
                equal to 'names', the the elements are interpreted as random
                variable names. If `None`, then the value of `dist._rv_mode` is
                consulted, which defaults to 'indices'.

            Returns
            -------
            result : RateDistortionResult
                The optimal rate-distortion pair.
            """
            rd = cls(dist, beta=beta, rv=rv, crvs=crvs, bound=bound, rv_mode=rv_mode)
            rd.optimize()
            pmf = rd.construct_joint(rd._optima)
            result = RateDistortionResult(rd.rate(pmf), rd.distortion(pmf))
            return result

        return rate_distortion


class RateDistortionHamming(BaseRateDistortion):
    """
    Rate distortion optimizer based around the Hamming distortion.
    """
    _optimization_backend = BaseRateDistortion._optimize_shotgun

    def _distortion(self):
        """
        Construct the distortion function.

        Returns
        -------
        distortion : func
            The distortion function.
        """
        hamming = 1 - np.eye(self._shape[0], self._shape[2])
        idx_xt = tuple(self._all_vars - (self._x | self._t))

        def distortion(pmf):
            """
            The Hamming distortion.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            d : float
                The average distortion.
            """
            pmf_xt = pmf.sum(axis=idx_xt)
            d = (hamming * pmf_xt).sum()
            return d

        return distortion


class RateDistortionResidualEntropy(BaseRateDistortion):
    """
    Rate distortion optimizer based around the residual entropy distortion.
    """
    def _distortion(self):
        """
        Construct the distortion function.

        Returns
        -------
        distortion : func
            The distortion function.
        """
        h = self._entropy(self._x | self._t, self._z)
        i = self._conditional_mutual_information(self._x, self._t, self._z)

        def distortion(pmf):
            """
            The residual entropy distortion.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            d : float
                The average distortion.
            """
            return h(pmf) - i(pmf)

        return distortion


class RateDistortionMaximumCorrelation(BaseRateDistortion):
    """
    Rate distortion optimizer based around the maximum correlation distortion.
    """
    def _distortion(self):
        """
        Construct the distortion function.

        Returns
        -------
        distortion : func
            The distortion function.
        """
        if self._shape[1] == 1:
            mc = self._maximum_correlation(self._x, self._t)
        else:
            mc = self._conditional_maximum_correlation(self._x, self._t, self._z)

        def distortion(pmf):
            """
            The maximum correlation distortion.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            d : float
                The average distortion.
            """
            return 1 - mc(pmf)

        return distortion
