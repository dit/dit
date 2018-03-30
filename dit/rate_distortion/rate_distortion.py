"""
"""

from abc import abstractmethod
from collections import namedtuple

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException
from ..multivariate import entropy
from ..utils import flatten


RateDistortionResult = namedtuple('RateDistortionResult', ['rate', 'distortion'])


class BaseRateDistortion(BaseAuxVarOptimizer):
    """
    Base optimizer for rate distortion type calculations.
    """
    _shotgun = 10

    def __init__(self, dist, beta=0.0, rv=None, crvs=None, bound=None, rv_mode=None):
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

        try:
            iter(rv[0])
            msg = "Rate-Distortion is only defined for one variable."
            raise ditException(msg)
        except:
            pass

        super(BaseRateDistortion, self).__init__(dist=dist, rvs=[rv], crvs=crvs, rv_mode=rv_mode)
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

        self._default_hops *= 2

    def _objective(self):
        """
        """
        def objective(self, x):
            """
            """
            pmf = self.construct_joint(x)
            obj = self.rate(pmf) + self._beta * self.distortion(pmf)
            return obj

        return objective

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
        """
        def rate_distortion(dist, beta=0.0, rv=None, crvs=None, bound=None, rv_mode=None):
            """
            """
            rd = cls(dist, beta=beta, rv=rv, crvs=crvs, bound=bound, rv_mode=rv_mode)
            rd.optimize()
            pmf = rd.construct_joint(rd._optima)
            result = RateDistortionResult(rd.rate(pmf), rd.distortion(pmf))
            return result

        return rate_distortion


class RateDistortionHamming(BaseRateDistortion):
    """
    """
    _type = "Hamming"
    _optimization_backend = BaseRateDistortion._optimize_shotgun

    def _distortion(self):
        """
        """
        hamming = 1 - np.eye(self._shape[0], self._shape[2])
        idx_xt = tuple(self._all_vars - (self._x | self._t))

        def distortion(pmf):
            pmf_xt = pmf.sum(axis=idx_xt)
            d = (hamming * pmf_xt).sum()
            return d

        return distortion


class RateDistortionTotalVariation(BaseRateDistortion):
    """
    """
    _type = "Total Variation"

    def _distortion(self):
        """
        """
        tv = self._total_variation(self._x, self._t)

        def distortion(pmf):
            """
            """
            return tv(pmf)

        return distortion


class RateDistortionResidualEntropy(BaseRateDistortion):
    """
    """
    _type = "Residual Entropy"

    def _distortion(self):
        """
        """
        h = self._entropy(self._x | self._t, self._z)
        i = self._conditional_mutual_information(self._x, self._t, self._z)

        def distortion(pmf):
            """
            """
            return h(pmf) - i(pmf)

        return distortion


class RateDistortionMaximumCorrelation(BaseRateDistortion):
    """
    """
    _type = "Maximum Correlation"

    def _distortion(self):
        """
        """
        if self._shape[1] == 1:
            mc = self._maximum_correlation(self._x, self._t)
        else:
            mc = self._conditional_maximum_correlation(self._x, self._t, self._z)

        def distortion(pmf):
            """
            """
            return 1 - mc(pmf)

        return distortion


class RDCurve(object):
    """
    """
    def __init__(self, dist, rv=None, crvs=None, beta_min=0, beta_max=10, beta_num=101, rd=RateDistortionHamming):
        """
        """
        if rv is None:
            rv = list(flatten(dist.rvs))

        self.dist = dist.copy()
        self.rv = rv
        self.crvs = crvs

        self.betas = np.linspace(beta_min, beta_max, beta_num)
        self._rd = rd

        d = dist.coalesce([self.rv])
        self._max_rate = entropy(d)
        rd = self._rd(d, beta=0.0)
        rd.optimize()
        self._max_distortion = rd.distortion(rd.construct_joint(rd._optima))

        try:
            dist_name = dist.name
        except AttributeError:
            dist_name = dist.__name__

        self.label = "{} {}".format(dist_name, rd._type)

        self.compute()

    def compute(self):
        """
        """
        rates = []
        distortions = []

        for beta in self.betas:
            rd = self._rd(self.dist, beta=beta, rv=self.rv, crvs=self.crvs)
            rd.optimize()
            pmf = rd.construct_joint(rd._optima)
            rates.append(rd.rate(pmf))
            distortions.append(rd.distortion(pmf))

        self.rates = np.asarray(rates)
        self.distortions = np.asarray(distortions)

    def plot(self, downsample=5):
        """
        """
        from .plotting import RDPlotter
        plotter = RDPlotter(self)
        return plotter.plot(downsample)

    def __add__(self, other):
        """
        """
        from .plotting import RDPlotter
        if isinstance(other, RDCurve):
            plotter = RDPlotter(self, other)
            return plotter
        else:
            return NotImplemented
