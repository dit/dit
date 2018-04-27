"""
Optimizers for computing information bottleneck points.
"""

from __future__ import division

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..divergences.pmf import relative_entropy
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
        alpha : float
            The alpha value for the generalized problem. alpha = 1.0 corresponds
            to the standard bottleneck, and alpha = 0.0 corresponds to the determinstic
            bottleneck.
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
        self.distortion = self._distortion()

        self._default_hops *= 3

    def _distortion(self):
        """
        Construct the distortion function.

        Returns
        -------
        distortion : func
            The distortion function.
        """
        cmi = self._conditional_mutual_information(self._x, self._y, self._z)(self.construct_joint(self.construct_random_initial()))
        relevance = self._conditional_mutual_information(self._y, self._t, self._z)

        def distortion(pmf):
            """
            Compute the distortion.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability mass function.

            Returns
            -------
            dist : float
                The average distortion value.
            """
            return cmi - relevance(pmf)

        return distortion

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
            obj = self.complexity(pmf) + self._beta * self.distortion(pmf)
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
            obj = self.entropy(pmf) - self._alpha * self.other(pmf) + self._beta * self.distortion(pmf)
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
            obj = self.entropy(pmf) + self._beta * self.distortion(pmf)
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
        Return a function which computes the result of this optimization.

        Returns
        -------
        information_bottleneck : func
            The function which performs this optimization.
        """
        def information_bottleneck(dist, beta, alpha=1.0, rvs=None, crvs=None, bound=None, rv_mode=None):
            """
            Compute an information bottleneck point.

            Parameters
            ----------
            dist : Distribution
                The distribution of interest.
            beta : float
                The beta value used in the objective function.
            alpha : float
                The alpha value for the generalized problem. alpha = 1.0 corresponds
                to the standard bottleneck, and alpha = 0.0 corresponds to the determinstic
                bottleneck.
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
            ib = cls(dist=dist,
                     beta=beta,
                     alpha=alpha,
                     rvs=rvs,
                     crvs=crvs,
                     bound=bound,
                     rv_mode=rv_mode,
                     )
            ib.optimize()
            pmf = ib.construct_joint(ib._optima)
            complexity = ib.complexity(pmf)
            relevance = ib.relevance(pmf)
            return complexity, relevance

        return information_bottleneck


class InformationBottleneckDivergence(InformationBottleneck):
    """
    A generalized information bottleneck which uses a distortion equal to
    D( p(Y|x) || q(Y|t) ) for an arbitrary divergence measure D.
    """
    def __init__(self, dist, beta, alpha=1.0, divergence=relative_entropy, rvs=None, crvs=None, bound=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        beta : float
            The beta value used in the objective function.
        alpha : float
            The alpha value for the generalized problem. alpha = 1.0 corresponds
            to the standard bottleneck, and alpha = 0.0 corresponds to the determinstic
            bottleneck.
        divergence : func
            The divergence to construct a bottleneck-like distortion measure from.
            Defaults to the relative entropy.
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
        self._divergence = divergence
        super(InformationBottleneckDivergence, self).__init__(dist=dist,
                                                              beta=beta,
                                                              alpha=alpha,
                                                              rvs=rvs,
                                                              crvs=crvs,
                                                              bound=bound,
                                                              rv_mode=rv_mode,
                                                              )
        self._default_hops *= 2

    def _distortion(self):
        """
        Construct the distortion measure from a divergence.

        Parameters
        ----------
        divergence : func
            A divergence measure.

        Returns
        -------
        distortion : func
            A function computing the average distortion.
        """

        if self._shape[2] > 1:
            idx_xyz = (3,)
            idx_yzt = (0,)
            idx_xzt = (1,)

            def distortion(pmf):
                """
                Compute the distortion.

                Parameters
                ----------
                pmf : np.ndarray
                    The joint probability mass function.
                """
                q_zxt = np.transpose(pmf.sum(axis=idx_xzt), (1, 0, 2))

                p_zxy = np.transpose(pmf.sum(axis=idx_xyz), (2, 0, 1))
                q_zty = np.transpose(pmf.sum(axis=idx_yzt), (1, 2, 0))
                p_y_zx = p_zxy / p_zxy.sum(axis=2, keepdims=True)
                q_y_zt = q_zty / q_zty.sum(axis=2, keepdims=True)

                dist_zxt = np.asarray([[[self._divergence(a, b) for b in q_y_t] for a in p_y_x] for p_y_x, q_y_t in zip(p_y_zx, q_y_zt)])
                dist_zxt[np.isinf(dist_zxt)] = 0
                dist = (q_zxt * dist_zxt).sum()
                return dist

        else:
            idx_xy = (2, 3)
            idx_yt = (0, 2)
            idx_xt = (1, 2)

            def distortion(pmf):
                """
                Compute the distortion.

                Parameters
                ----------
                pmf : np.ndarray
                    The joint probability mass function.
                """
                q_xt = pmf.sum(axis=idx_xt)

                p_xy = pmf.sum(axis=idx_xy)
                q_ty = pmf.sum(axis=idx_yt).T
                p_y_x = p_xy / p_xy.sum(axis=1, keepdims=True)
                q_y_t = q_ty / q_ty.sum(axis=1, keepdims=True)

                dist_xt = np.asarray([[self._divergence(a, b) for b in q_y_t] for a in p_y_x])
                dist_xt[np.isinf(dist_xt)] = 0
                dist = (q_xt * dist_xt).sum()

                return dist

        return distortion
