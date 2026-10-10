"""
Optimizers for computing information bottleneck points.
"""

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..divergences.pmf import relative_entropy
from ..exceptions import ditException

__all__ = (
    "DeterministicInformationBottleneck",
    "GeneralizedInformationBottleneck",
    "InformationBottleneck",
    "InformationBottleneckDivergence",
)


class InformationBottleneck(BaseAuxVarOptimizer):
    """
    Base optimizer for information bottleneck type calculations.
    """

    _shotgun = 10

    def __init__(self, dist, beta, alpha=1.0, rvs=None, crvs=None, bound=None):
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
        """
        if rvs is None:
            rvs = dist.rvs

        if len(rvs) != 2:
            msg = "The information bottleneck is only defined for two variables."
            raise ditException(msg)

        super().__init__(dist=dist, rvs=rvs, crvs=crvs)
        if not 0.0 <= alpha <= 1.0:
            msg = "alpha must be in [0.0, 1.0]."
            raise ditException(msg)
        else:
            self._alpha = alpha
        if not beta >= 0.0:
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

        self._construct_auxvars([(self._x | self._z, bound)])

        self.complexity = self._conditional_mutual_information(self._x, self._t, self._z)
        self.entropy = self._entropy(self._t, self._z)
        self.relevance = self._conditional_mutual_information(self._y, self._t, self._z)
        self.other = self._entropy(self._t, self._x | self._z)
        self.error = self._conditional_mutual_information(self._x, self._y, self._t | self._z)
        self.distortion = self._distortion()

        self._default_hops *= 3

    _max_partition_seeds = 2000
    _n_fixed_point_seeds = 10

    def _partition_encoders(self):
        """
        Deterministic encoders T = f(X), one per partition of X into at most
        ``bound`` blocks, when Z is trivial and there are few enough of them.
        For the deterministic bottleneck (alpha = 0) the objective is concave in
        the encoder, so its minimum is among these.
        """
        n_x, n_z, bound = self._aux_vars[0].shape
        if n_z != 1:
            return []
        labels = [[0]]
        for _ in range(n_x - 1):
            labels = [lab + [k] for lab in labels for k in range(min(max(lab) + 2, bound))]
            if len(labels) > self._max_partition_seeds:
                return []
        encoders = []
        for lab in labels:
            enc = np.zeros((n_x, 1, bound))
            enc[np.arange(n_x), 0, lab] = 1
            encoders.append(enc)
        return encoders

    def _fixed_point(self, enc, iters=500, tol=1e-10):
        """
        Iterate the information bottleneck self-consistent equations
        :cite:`tishby2000information`, p(t|x) ∝ p(t) exp(-beta KL[p(y|x) || p(y|t)]),
        separately for each value of Z.
        """
        enc = enc.copy()
        for z in range(enc.shape[1]):
            p_xy = self._pmf[:, :, z]
            p_x = p_xy.sum(axis=1)
            if p_x.sum() <= 0:
                continue
            keep = p_x > 0
            p_y_x = p_xy[keep] / p_x[keep, None]
            e = enc[keep, z, :]
            for _ in range(iters):
                p_t = p_x[keep] @ e
                p_y_t = (p_xy[keep].T @ e) / np.maximum(p_t, 1e-300)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(
                        p_y_x[:, :, None] > 0, np.log(p_y_x[:, :, None] / np.maximum(p_y_t[None], 1e-300)), 0.0
                    )
                kl = (p_y_x[:, :, None] * ratio).sum(axis=1)
                new = p_t[None, :] * np.exp(-self._beta * (kl - kl.min(axis=1, keepdims=True)))
                new /= new.sum(axis=1, keepdims=True)
                done = np.abs(new - e).max() < tol
                e = new
                if done:
                    break
            enc[keep, z, :] = e
        return enc

    def _seed_initials(self):
        """
        The constant and identity encoders and, for the standard bottleneck,
        fixed points of the self-consistent equations reached from random encoders.

        Returns
        -------
        seeds : list of np.ndarray
            Optimization vectors.
        """
        shape = self._aux_vars[0].shape
        seeds = [self.construct_constant_initial(), self.construct_copy_initial()]
        if np.isclose(self._alpha, 1.0):
            rng = np.random.default_rng() if self._rng is None else self._rng
            for _ in range(self._n_fixed_point_seeds):
                enc = rng.dirichlet(np.ones(shape[-1]), size=shape[:-1])
                seeds.append(self._fixed_point(enc).ravel())
        return seeds

    def optimize(self, *args, **kwargs):
        """
        Optimize, then keep the best of the result, the seeds, and the
        deterministic partition encoders, since local search started at a seed
        can drift away from it.

        Parameters
        ----------
        args, kwargs
            Passed to :meth:`BaseAuxVarOptimizer.optimize`.

        Returns
        -------
        result : OptimizeResult
            The result of the search.
        """
        result = super().optimize(*args, **kwargs)
        partitions = [enc.ravel() for enc in self._partition_encoders()]
        self._optima = min([self._optima, *self._seed_initials(), *partitions], key=self.objective).copy()
        return result

    def _distortion(self):
        """
        Construct the distortion function.

        Returns
        -------
        distortion : func
            The distortion function.
        """
        cmi = self._conditional_mutual_information(self._x, self._y, self._z)(
            self.construct_joint(self.construct_random_initial())
        )
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

    def _objective_gradient(self):
        """
        Analytic gradient of the information-bottleneck objective w.r.t. the
        joint, matching the alpha-branch selected by :meth:`_objective`.

        The distortion ``I[X:Y|Z] - I[Y:T|Z]`` has a constant first term, so its
        gradient is ``-grad I[Y:T|Z]``.
        """
        beta = self._beta
        relevance_g = self._conditional_mutual_information_grad(self._y, self._t, self._z)

        def distortion_g(pmf):
            return -relevance_g(pmf)

        if np.isclose(self._alpha, 1.0):
            complexity_g = self._conditional_mutual_information_grad(self._x, self._t, self._z)

            def grad(pmf):
                return complexity_g(pmf) + beta * distortion_g(pmf)

        elif np.isclose(self._alpha, 0.0):
            entropy_g = self._entropy_grad(self._t, self._z)

            def grad(pmf):
                return entropy_g(pmf) + beta * distortion_g(pmf)

        else:
            entropy_g = self._entropy_grad(self._t, self._z)
            other_g = self._entropy_grad(self._t, self._x | self._z)
            alpha = self._alpha

            def grad(pmf):
                return entropy_g(pmf) - alpha * other_g(pmf) + beta * distortion_g(pmf)

        return grad

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

        def information_bottleneck(dist, beta, alpha=1.0, rvs=None, crvs=None, bound=None):
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
            """
            ib = cls(
                dist=dist,
                beta=beta,
                alpha=alpha,
                rvs=rvs,
                crvs=crvs,
                bound=bound,
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

    def __init__(self, dist, beta, alpha=1.0, divergence=relative_entropy, rvs=None, crvs=None, bound=None):
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
        """
        self._divergence = divergence
        super().__init__(
            dist=dist,
            beta=beta,
            alpha=alpha,
            rvs=rvs,
            crvs=crvs,
            bound=bound,
        )
        self._default_hops *= 2

    # The general-divergence distortion is not covered by the standard
    # gradient builders, so fall back to finite differences here.
    _objective_gradient = None

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

                dist_zxt = np.asarray(
                    [
                        [[self._divergence(a, b) for b in q_y_t] for a in p_y_x]
                        for p_y_x, q_y_t in zip(p_y_zx, q_y_zt, strict=True)
                    ]
                )
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


class GeneralizedInformationBottleneck(InformationBottleneck):
    """
    The generalized information bottleneck family.

    The parameter ``alpha`` interpolates between the standard information
    bottleneck (``alpha=1``) and the deterministic information bottleneck
    (``alpha=0``).
    """


class DeterministicInformationBottleneck(GeneralizedInformationBottleneck):
    """
    The deterministic information bottleneck objective.

    This is the ``alpha=0`` endpoint of the generalized information bottleneck,
    using :math:`H(T | Z)` as the compression term.
    """

    def __init__(self, dist, beta, alpha=0.0, rvs=None, crvs=None, bound=None):
        """
        Initialize the deterministic bottleneck optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        beta : float
            The beta value used in the objective function.
        alpha : float
            Must be 0.0. Accepted only so inherited helpers can instantiate this
            class through the same call signature as InformationBottleneck.
        rvs : iter of iters, None
            The random variables to compute the bottleneck of.
        crvs : iter, None
            The random variables to condition on.
        bound : int, None
            The bound on the size of the statistic. If None, use the size of X.
        """
        if not np.isclose(alpha, 0.0):
            msg = "DeterministicInformationBottleneck requires alpha=0.0."
            raise ditException(msg)
        super().__init__(
            dist=dist,
            beta=beta,
            alpha=0.0,
            rvs=rvs,
            crvs=crvs,
            bound=bound,
        )
