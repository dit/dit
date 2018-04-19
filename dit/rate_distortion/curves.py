"""
Objects to compute single rate-distortion curves.
"""

from __future__ import division

import numpy as np

from .blahut_arimoto import blahut_arimoto, blahut_arimoto_ib
from .distortions import hamming
from .information_bottleneck import InformationBottleneck, InformationBottleneckDivergence
from .. import Distribution
from ..algorithms.minimal_sufficient_statistic import mss
from ..exceptions import ditException
from ..multivariate import entropy, total_correlation
from ..utils import flatten


class RDCurve(object):
    """
    Compute a rate-distortion curve.
    """

    def __init__(self, dist, rv=None, crvs=None, beta_min=0, beta_max=10, beta_num=101, alpha=1.0, distortion=hamming, method='sp'):
        """
        Initialize the curve computer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rv : iterable, None
            The random variables to compute the rate-distortion curve of.
            If None, use all.
        crvs : iterable, None
            The random variables to condition on.
        beta_min : float
            The minimum beta value for the curve. Defaults to 0.
        beta_max : float
            The maximum beta value for the curve. Defaults to 10. If None,
            iteratively find a beta value with nearly maximal rate.
        beta_num : int
            The number of beta values for the curve. Defaults to 101.
        alpha : float
            The alpha value to utilize. 1.0 corresponds to the standard information
            bottleneck, while 0.0 corresponds to the deterministic bottleneck.
        distortion : Distortion
            The distortion to use.
        method : {'sp', 'ba'}
            The method to utilize in computing the curve. If 'sp', utilize
            scipy.optimize; if 'ba' utilize the iterative Blahut-Arimoto
            algorithm. Defaults to 'sp'.
        """
        if rv is None:
            rv = list(flatten(dist.rvs))

        self.dist = dist.copy()
        self.rv = rv
        self.crvs = crvs

        d = dist.coalesce([self.rv])
        self.p_x = d.pmf

        self._distortion = distortion

        if method not in ('sp', 'ba'):  # pragma: no cover
            msg = "Method '{}' not supported.".format(method)
            raise ditException(msg)
        elif method == 'sp' and not distortion.optimizer:  # pragma: no cover
            msg = "Method is 'sp' but distortion does not have an optimizer."
            raise ditException(msg)
        elif method == 'ba' and not distortion.matrix:  # pragma: no cover
            msg = "Method is 'ba' but distortion does not have a matrix."
            raise ditException(msg)
        elif method == 'ba' and crvs:  # pragma: no cover
            msg = "Method 'ba' does not support conditional variables."
            raise ditException(msg)
        else:
            self._get_rd = {'ba': self._get_rd_ba,
                            'sp': self._get_rd_sp,
                            }[method]

        self._rd_opt = self._distortion.optimizer(self.dist,
                                                  beta=0.0,
                                                  alpha=alpha,
                                                  rv=self.rv,
                                                  crvs=self.crvs)

        self._max_rate = entropy(d)
        _, self._max_distortion, _, _ = self._get_rd(beta=0.0)
        self._max_rank = len(d.outcomes)

        if beta_max is None:
            beta_max = self.find_max_beta()
        self.betas = np.linspace(beta_min, beta_max, beta_num)

        try:  # pragma: no cover
            dist_name = [dist.name]
        except AttributeError:
            dist_name = []
        self.label = " ".join(dist_name + [self._distortion.name])

        self.compute()

    def __add__(self, other):  # pragma: no cover
        """
        Combine two RDCurves into an RDPlotter.

        Parameters
        ----------
        other : RDCurve
            The curve to aggregate with `self`.

        Returns
        -------
        plotter : RDPlotter
            A plotter with both `self` and `other`.
        """
        from .plotting import RDPlotter
        if isinstance(other, RDCurve):
            plotter = RDPlotter(self, other)
            return plotter
        else:
            return NotImplemented

    def find_max_beta(self):
        """
        Find a beta value which maximizes the rate.

        Returns
        -------
        beta_max : float
            The the smallest found beta value which achieves minimal
            distortion.
        """
        beta_max = 1
        rate = 0

        while not np.isclose(rate, self._max_rate, atol=1e-5, rtol=1e-5):
            beta_max = 1.5*beta_max
            rate, _, _, _ = self._get_rd(beta=beta_max)

        return beta_max

    def _get_rd_sp(self, beta, initial=None):
        """
        Compute the rate-distortion pair for `beta` using scipy.optimize.

        Parameters
        ----------
        beta : float
            The beta value to optimize for.
        initial : np.ndarray, None
            An initial optimization vector, useful for numerical continuation.

        Returns
        -------
        r : float
            The rate.
        d : float
            The distortion.
        q : np.ndarray
            The matrix p(x, x_hat)
        x0 : np.ndarray
            The found optima.
        """
        self._rd_opt._beta = beta
        self._rd_opt.optimize(x0=initial)
        x0 = self._rd_opt._optima.copy()
        q = self._rd_opt.construct_joint(self._rd_opt._optima)
        r = self._rd_opt.rate(q)
        d = self._rd_opt.distortion(q)
        return r, d, q.sum(axis=1), x0

    def _get_rd_ba(self, beta, initial=None):
        """
        Compute the rate-distortion pair for `beta` using Blahut-Arimoto.

        Parameters
        ----------
        beta : float
            The beta value to optimize for.
        initial : np.ndarray, None
            An initial optimization vector, useful for numerical continuation.

        Returns
        -------
        r : float
            The rate.
        d : float
            The distortion.
        q : np.ndarray
            The matrix p(x, x_hat)
        x0 : np.ndarray
            The found optima.
        """
        (r, d), q = blahut_arimoto(p_x=self.p_x,
                                   beta=beta,
                                   distortion=self._distortion.matrix,
                                   )
        return r, d, q, initial

    def compute(self):
        """
        Sweep beta and compute the rate-distortion curve.

        Parameters
        ----------
        method : {'sp', 'ba'}
            The method of computation to use. 'sp' denotes scipy.optimize;
            'ba' denotes blahut-arimoto.
        """
        rates = []
        distortions = []
        ranks = []
        alphabets = []

        x0 = None

        for beta in self.betas[::-1]:
            r, d, q, x0 = self._get_rd(beta, initial=x0)
            rates.append(r)
            distortions.append(d)

            q_x_xhat = q / q.sum(axis=0, keepdims=True)

            ranks.append(np.linalg.matrix_rank(q_x_xhat, tol=1e-5))
            alphabets.append((q.sum(axis=0) > 1e-6).sum())

        self.rates = np.asarray(rates)[::-1]
        self.distortions = np.asarray(distortions)[::-1]
        self.ranks = np.asarray(ranks)[::-1]
        self.alphabets = np.asarray(alphabets)[::-1]

    def plot(self, downsample=5):  # pragma: no cover
        """
        Construct an RDPlotter and utilize it to plot the rate-distortion
        curve.

        Parameters
        ----------
        downsample : int
            The how frequent to display points along the RD curve.

        Returns
        -------
        fig : plt.figure
            The resulting figure.
        """
        from .plotting import RDPlotter
        plotter = RDPlotter(self)
        return plotter.plot(downsample)


class IBCurve(object):
    """
    Compute an information bottleneck curve.
    """

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None, beta_min=0.0, beta_max=15.0, beta_num=101, alpha=1.0, method='sp', divergence=None):
        """
        Initialize the curve computer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rv : iterable, None
            The random variables to compute the information bottleneck curve of.
            If None, use [0], [1].
        crvs : iterable, None
            The random variables to condition on.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        beta_min : float
            The minimum beta value for the curve. Defaults to 0.
        beta_max : float, None
            The maximum beta value for the curve. Defaults to 15. If None,
            iteratively find a beta value with nearly maximal complexity.
        beta_num : int
            The number of beta values for the curve. Defaults to 101.
        alpha : float
            The alpha value to utilize. 1.0 corresponds to the standard information
            bottleneck, while 0.0 corresponds to the deterministic bottleneck.
        method : {'sp', 'ba'}
            The method to utilize in computing the curve. If 'sp', utilize
            scipy.optimize; if 'ba' utilize the iterative Blahut-Arimoto
            algorithm. Defaults to 'sp'.
        divergence : func
            The divergence measure to use as a distortion. Defaults to the standard
            relative entropy.
        """
        self.dist = dist.copy()
        self.dist.make_dense()

        self._x, self._y = rvs if rvs is not None else ([0], [1])
        self._z = crvs if crvs is not None else []
        self._aux = [dist.outcome_length()]
        self._rv_mode = rv_mode

        self.p_xy = self.dist.coalesce([self._x, self._y])
        self.p_xy = self.p_xy.pmf.reshape(tuple(map(len, self.p_xy.alphabet)))

        args = {'dist': self.dist,
                'beta': 0.0,
                'alpha': alpha,
                'rvs': [self._x, self._y],
                'crvs': self._z,
                'rv_mode': self._rv_mode
        }

        if divergence is not None:
            bottleneck = InformationBottleneckDivergence
            args['divergence'] = divergence
        else:
            bottleneck = InformationBottleneck
        self._bn = bottleneck(**args)

        self._max_complexity = entropy(mss(dist, self._x, self._y))
        self._max_relevance = total_correlation(dist, [self._x, self._y])
        self._max_rank = len(dist.marginal(self._x).outcomes)
        self._max_distortion = self._bn.distortion(self._get_opt_sp(beta=0.0)[0])

        if np.isclose(alpha, 1.0):
            self.label = "IB"
        elif np.isclose(alpha, 0.0):
            self.label = "DIB"
        else:
            self.label = "GIB({:.3f})".format(alpha)

        beta_max = self.find_max_beta() if beta_max is None else beta_max
        self.betas = np.linspace(beta_min, beta_max, beta_num)

        self.compute(method)

    def __add__(self, other):  # pragma: no cover
        """
        Combine two IBCurves into an IBPlotter.

        Parameters
        ----------
        other : IBCurve
            The curve to aggregate with `self`.

        Returns
        -------
        plotter : IBPlotter
            A plotter with both `self` and `other`.
        """
        from .plotting import IBPlotter
        if isinstance(other, IBCurve):
            plotter = IBPlotter(self, other)
            return plotter
        else:
            return NotImplemented

    def _get_opt_sp(self, beta, initial=None):
        """
        Compute the information bottleneck solution for `beta` using scipy.optimize.

        Parameters
        ----------
        beta : float
            The beta value to optimize for.
        initial : np.ndarray, None
            An initial optimization vector, useful for numerical continuation.

        Returns
        -------
        q : np.ndarray
            The matrix p(x, y, z, t)
        x0 : np.ndarray
            The found optima.
        """
        self._bn._beta = beta
        self._bn.optimize(x0=initial)
        x0 = self._bn._optima.copy()
        q_xyzt = self._bn.construct_joint(self._bn._optima)
        return q_xyzt, x0

    def _get_opt_ba(self, beta, initial=None):  # pragma: no cover
        """
        Compute the information bottleneck solution for `beta` using blahut-arimoto.

        Parameters
        ----------
        beta : float
            The beta value to optimize for.
        initial : np.ndarray, None
            An initial optimization vector, useful for numerical continuation.

        Returns
        -------
        q : np.ndarray
            The matrix p(x, y, z, t)
        x0 : np.ndarray
            The found optima.
        """
        q_xyt = blahut_arimoto_ib(p_xy=self.p_xy, beta=beta)[1]
        q_xyzt = q_xyt[:, :, np.newaxis, :]
        return q_xyzt, None

    def compute(self, method='sp'):
        """
        Sweep beta and compute the information bottleneck curve.

        Parameters
        ----------
        method : {'sp', 'ba'}
            The method of computation to use. 'sp' denotes scipy.optimize;
            'ba' denotes blahut-arimoto.
        """
        get_opt = {'ba': self._get_opt_ba,
                   'sp': self._get_opt_sp,
                   }[method]

        complexities = []
        entropies = []
        relevances = []
        errors = []
        ranks = []
        alphabets = []
        distortions = []

        x, y, z, t = [[0], [1], [2], [3]]

        x0 = None

        for beta in self.betas[::-1]:
            q_xyzt, x0 = get_opt(beta, x0)
            d = Distribution.from_ndarray(q_xyzt)
            complexities.append(total_correlation(d, [x, t], z))
            entropies.append(entropy(d, x, z))
            relevances.append(total_correlation(d, [y, t], z))
            errors.append(total_correlation(d, [x, y], z + t))
            distortions.append(self._bn.distortion(q_xyzt))

            q_xt = q_xyzt.sum(axis=(1, 2))
            q_x_t = (q_xt / q_xt.sum(axis=0, keepdims=True))

            ranks.append(np.linalg.matrix_rank(q_x_t, tol=1e-4))
            alphabets.append((q_xt.sum(axis=0) > 1e-6).sum())

        self.complexities = np.asarray(complexities)[::-1]
        self.entropies = np.asarray(entropies)[::-1]
        self.relevances = np.asarray(relevances)[::-1]
        self.errors = np.asarray(errors)[::-1]
        self.ranks = np.asarray(ranks)[::-1]
        self.alphabets = np.asarray(alphabets)[::-1]
        self.distortions = np.asarray(distortions)[::-1]

    def find_max_beta(self):
        """
        Find a beta value which maximizes the rate.

        Returns
        -------
        beta_max : float
            The the smallest found beta value which achieves minimal
            distortion.
        """
        beta_max = 1.0
        relevance = 0.0

        while not np.isclose(relevance, self._max_relevance, atol=1e-5, rtol=1e-5):
            beta_max = 1.5*beta_max
            q, _ = self._get_opt_sp(beta=beta_max)
            relevance = self._bn.relevance(q)

        return beta_max

    def find_kinks(self):
        """
        Determine the beta values where new features are discovered.

        Returns
        -------
        kinks : np.ndarray
            An array of beta values where new features are discovered.
        """
        diff = np.diff(self.ranks)
        jumps = np.arange(len(diff))[diff > 0]
        kinks = np.asarray([jump for jump in jumps if diff[jump-1] == 0])
        return self.betas[kinks]

    def plot(self, downsample=5):  # pragma: no cover
        """
        Construct an IBPlotter and utilize it to plot the information
        bottleneck curve.

        Parameters
        ----------
        downsample : int
            The how frequent to display points along the IB curve.

        Returns
        -------
        fig : plt.figure
            The resulting figure.
        """
        from .plotting import IBPlotter
        plotter = IBPlotter(self)
        return plotter.plot(downsample)
