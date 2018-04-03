"""
"""

from __future__ import division

import numpy as np

from .blahut_arimoto import blahut_arimoto, blahut_arimoto_ib
from .distortions import hamming
from .information_bottleneck import (DeterministicInformationBottleneck,
                                     GeneralizedInformationBottleneck,
                                     InformationBottleneck,
                                     )
from .. import Distribution
from ..multivariate import entropy, total_correlation
from ..utils import flatten


class RDCurve(object):
    """
    """
    def __init__(self, dist, rv=None, crvs=None, beta_min=0, beta_max=10, beta_num=101, distortion=hamming, ba=False):
        """
        """
        if rv is None:
            rv = list(flatten(dist.rvs))

        self.dist = dist.copy()
        self.rv = rv
        self.crvs = crvs

        self.betas = np.linspace(beta_min, beta_max, beta_num)
        self._distortion = distortion

        d = dist.coalesce([self.rv])
        self._max_rate = entropy(d)
        rd = self._distortion.optimizer(d, beta=0.0)
        rd.optimize()
        self._max_distortion = rd.distortion(rd.construct_joint(rd._optima))

        try:
            dist_name = dist.name
        except AttributeError:
            dist_name = r"\b"

        self.label = "{} {}".format(dist_name, rd._type)

        if ba:
            self.compute_ba()
        else:
            self.compute_sp()

    def compute_sp(self):
        """
        """
        rates = []
        distortions = []

        for beta in self.betas:
            rd = self._distortion.optimizer(self.dist, beta=beta, rv=self.rv, crvs=self.crvs)
            rd.optimize()
            pmf = rd.construct_joint(rd._optima)
            rates.append(rd.rate(pmf))
            distortions.append(rd.distortion(pmf))

        self.rates = np.asarray(rates)
        self.distortions = np.asarray(distortions)

    def compute_ba(self):
        """
        """
        p_x = self.dist.marginal(self.rv).pmf

        rates = []
        distortions = []

        for beta in self.betas:
            r, d = blahut_arimoto(p_x=p_x,
                                  beta=beta,
                                  distortion=self._distortion.matrix,
                                  )
            rates.append(r)
            distortions.append(d)

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


class IBCurve(object):
    """
    """

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None, beta_min=0.0, beta_max=15.0, beta_num=101, alpha=1.0, ba=False):
        """
        """
        self.dist = dist.copy()
        self.dist.make_dense()

        self._x, self._y = rvs if rvs is not None else ([0], [1])
        self._z = crvs if crvs is not None else []
        self._aux = [dist.outcome_length()]
        self._rv_mode = rv_mode

        self._true_complexity = entropy(dist, self._x)  # TODO: replace with MSS
        self._true_relevance = total_correlation(dist, [self._x, self._y])

        self._args = {'dist': self.dist,
                      'rvs': [self._x, self._y],
                      'crvs': self._z,
                      'rv_mode': self._rv_mode,
                      }

        if np.isclose(alpha, 1.0):
            self._bottleneck = InformationBottleneck
            self.label = "IB"
        elif np.isclose(alpha, 0.0):
            self._bottleneck = DeterministicInformationBottleneck
            self.label = "DIB"
        else:
            self._bottleneck = GeneralizedInformationBottleneck
            self._args['alpha'] = alpha
            self.label = "GIB({:.3f})".format(alpha)

        beta_max = self.find_max_beta() if beta_max is None else beta_max
        self.betas = np.linspace(beta_min, beta_max, beta_num)

        self.compute_sp()

    def compute_sp(self):
        """
        """
        complexities = []
        entropies = []
        relevances = []
        errors = []
        ranks = []

        x0 = None

        for beta in self.betas:
            ib = self._bottleneck(beta=beta, **self._args)
            ib.optimize(x0=x0)
            x0 = ib._optima.copy()
            pmf = ib.construct_joint(ib._optima)
            complexities.append(ib.complexity(pmf))
            entropies.append(ib.entropy(pmf))
            relevances.append(ib.relevance(pmf))
            errors.append(ib.error(pmf))
            ranks.append(np.linalg.matrix_rank(pmf.sum(axis=1)))

        self.complexities = np.asarray(complexities)
        self.entropies = np.asarray(entropies)
        self.relevances = np.asarray(relevances)
        self.errors = np.asarray(errors)
        self.ranks = np.asarray(ranks)

    def compute_ba(self):
        """
        """
        complexities = []
        entropies = []
        relevances = []
        errors = []

        p_xy = self.dist.coalesce(self._x + self._y)
        p_xy = p_xy.pmf.reshape(tuple(map(len, p_xy.alphabet)))

        for beta in self.betas:
            q_xyt = blahut_arimoto_ib(p_xy=p_xy,
                                      beta=beta,
                                      )
            d = Distribution.from_ndarray(q_xyt)
            complexities.append(total_correlation(d, [[0], [2]]))
            entropies.append(entropy(d, [2]))
            relevances.append(total_correlation(d, [[1], [2]]))
            errors.append(total_correlation(d, [[0], [1]], [2]))

        self.complexities = np.asarray(complexities)
        self.entropies = np.asarray(entropies)
        self.relevances = np.asarray(relevances)
        self.errors = np.asarray(errors)

    def find_max_beta(self):
        """
        """
        beta_max = 2/3
        relevance = 0

        while not np.isclose(relevance, self._true_relevance, atol=1e-3, rtol=1e-3):
            beta_max = int(np.ceil(1.5*beta_max))
            ib = self._bottleneck(beta=beta_max, **self._args)
            ib.optimize()
            relevance = ib.relevance(ib.construct_joint(ib._optima))

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
