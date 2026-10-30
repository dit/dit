"""
Optimizers for computing information bottleneck points.
"""

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..divergences.pmf import relative_entropy
from ..exceptions import OptimizationException, ditException

__all__ = (
    "DeterministicInformationBottleneck",
    "GeneralizedInformationBottleneck",
    "InformationBottleneck",
    "InformationBottleneckDivergence",
    "InformationBottleneckRate",
    "information_bottleneck_rate",
)


def partition_encoders(shape, max_seeds):
    """
    Deterministic encoders T = f(X), one per partition of X into at most
    ``bound`` blocks, when Z is trivial and there are at most ``max_seeds`` of
    them.

    Parameters
    ----------
    shape : tuple
        The encoder shape ``(|X|, |Z|, bound)``.
    max_seeds : int
        The maximum number of partitions to enumerate.

    Returns
    -------
    encoders : list of np.ndarray
        The encoders, each of shape ``shape``; empty if Z is non-trivial or
        there are too many partitions.
    """
    n_x, n_z, bound = shape
    if n_z != 1:
        return []
    labels = [[0]]
    for _ in range(n_x - 1):
        labels = [lab + [k] for lab in labels for k in range(min(max(lab) + 2, bound))]
        if len(labels) > max_seeds:
            return []
    encoders = []
    for lab in labels:
        enc = np.zeros((n_x, 1, bound))
        enc[np.arange(n_x), 0, lab] = 1
        encoders.append(enc)
    return encoders


class InformationBottleneck(BaseAuxVarOptimizer):
    """
    Base optimizer for information bottleneck type calculations.
    """

    _shotgun = 10

    # The Lagrangian needs at most |X||Z| symbols; a rate constraint adds one more.
    _extra_symbols = 0

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

        tbound = self._shape[0] * self._shape[2] + self._extra_symbols
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
        return partition_encoders(self._aux_vars[0].shape, self._max_partition_seeds)

    def _fixed_point(self, enc, iters=500, tol=1e-10, beta=None):
        """
        Iterate the information bottleneck self-consistent equations
        :cite:`tishby2000information`, p(t|x) ∝ p(t) exp(-beta KL[p(y|x) || p(y|t)]),
        separately for each value of Z. ``beta`` defaults to the optimizer's.
        """
        beta = self._beta if beta is None else beta
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
                new = p_t[None, :] * np.exp(-beta * (kl - kl.min(axis=1, keepdims=True)))
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

        def information_bottleneck(
            dist, beta, alpha=1.0, rvs=None, crvs=None, bound=None, niter=None, maxiter=None, rng=None
        ):
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
            niter : int, None
                The number of basin hops to perform.
            maxiter : int, None
                The number of iterations of the optimization subroutine.
            rng : np.random.Generator, int, None
                Source of randomness for the optimizer.
            """
            ib = cls(
                dist=dist,
                beta=beta,
                alpha=alpha,
                rvs=rvs,
                crvs=crvs,
                bound=bound,
            )
            ib.optimize(niter=niter, maxiter=maxiter, rng=rng)
            pmf = ib.construct_joint(ib._optima)
            complexity = ib.complexity(pmf)
            relevance = ib.relevance(pmf)
            return complexity, relevance

        return information_bottleneck


class InformationBottleneckRate(InformationBottleneck):
    """
    The information bottleneck at a fixed rate :cite:`tishby2000information`,

    .. math::

        F(R) = \\max_{p(t | x, z) : I[X : T | Z] \\leq R} I[Y : T | Z],

    the information bottleneck curve as a function of the complexity. The curve
    is concave and non-decreasing, and reaches ``I[X : Y | Z]``. A rate
    constraint needs one symbol more than the Lagrangian, so the size of T is
    bounded by ``|X||Z| + 1``.
    """

    _extra_symbols = 1

    # SLSQP and polishing leave the optimum on the active constraint up to about
    # this much slack.
    _feasibility_tol = 1e-6

    # Independent basin-hopping searches; a single one occasionally ends in a
    # poor local maximum.
    _restarts = 3

    # Fixed points of the self-consistent equations at these betas lie near the
    # curve over a range of rates, and seed the search.
    _seed_betas = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 50.0, 100.0)

    def __init__(self, dist, rate, rvs=None, crvs=None, bound=None):
        """
        Initialize the bottleneck.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rate : float
            The maximum complexity R, in bits.
        rvs : iter of iters, None
            The variables X and Y. If None, use ``dist.rvs``.
        crvs : iter, None
            The random variables to condition on.
        bound : int, None
            The bound on the size of T. If None, use ``|X||Z| + 1``.
        """
        if not rate >= 0.0:
            msg = "rate must be non-negative."
            raise ditException(msg)
        self._rate = rate
        super().__init__(dist=dist, beta=0.0, rvs=rvs, crvs=crvs, bound=bound)

        complexity_g = self._conditional_mutual_information_grad(self._x, self._t, self._z)
        self.constraints += [
            {
                "type": "ineq",
                "fun": self._constraint_rate,
                "jac": lambda x: -self._construct_joint_vjp(x, np.asarray(complexity_g(self.construct_joint(x)))),
            },
        ]

    def _constraint_rate(self, x):
        """
        Non-negative when the complexity is within the rate.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        slack : float
            ``R - I[X : T | Z]``.
        """
        return self._rate - self.complexity(self.construct_joint(x))

    def _objective(self):
        """
        The negated relevance.

        Returns
        -------
        objective : func
            The objective function.
        """

        def objective(self, x):
            """
            Compute -I[Y : T | Z].

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            return -self.relevance(self.construct_joint(x))

        return objective

    def _objective_gradient(self):
        """
        Analytic gradient of the negated relevance w.r.t. the joint.
        """
        relevance_g = self._conditional_mutual_information_grad(self._y, self._t, self._z)
        return lambda pmf: -relevance_g(pmf)

    def _seed_initials(self):
        """
        The constant and identity encoders, and fixed points of the
        self-consistent equations over a range of betas.

        Returns
        -------
        seeds : list of np.ndarray
            Optimization vectors.
        """
        shape = self._aux_vars[0].shape
        rng = np.random.default_rng() if self._rng is None else self._rng
        seeds = [self.construct_constant_initial(), self.construct_copy_initial()]
        for beta in self._seed_betas:
            enc = rng.dirichlet(np.ones(shape[-1]), size=shape[:-1])
            seeds.append(self._fixed_point(enc, beta=beta).ravel())
        return seeds

    def optimize(self, *args, **kwargs):
        """
        Optimize a few times, then keep the most relevant feasible candidate
        among the results, the seeds, and the deterministic encoders.

        Parameters
        ----------
        args, kwargs
            Passed to :meth:`BaseAuxVarOptimizer.optimize`.

        Returns
        -------
        result : OptimizeResult
            The result of the search.
        """
        optima = []
        result = None
        rng = kwargs.get("rng")
        if rng is not None and not isinstance(rng, np.random.Generator):
            # One generator for all restarts, so that they differ.
            kwargs["rng"] = np.random.default_rng(rng)
        for _ in range(self._restarts):
            try:
                result = BaseAuxVarOptimizer.optimize(self, *args, **kwargs)
            except OptimizationException:
                continue
            optima.append(self._optima)
        candidates = [*optima, *self._seed_initials(), *(enc.ravel() for enc in self._partition_encoders())]
        feasible = [x for x in candidates if self._constraint_rate(x) >= -self._feasibility_tol]
        # The constant encoder is always feasible.
        self._optima = max(feasible, key=lambda x: self.relevance(self.construct_joint(x))).copy()
        return result

    @classmethod
    def functional(cls):
        """
        Return a function which computes the result of this optimization.

        Returns
        -------
        information_bottleneck_rate : func
            The function which performs this optimization.
        """

        def information_bottleneck_rate(
            dist, rate, rvs=None, crvs=None, bound=None, niter=None, maxiter=None, rng=None
        ):
            """
            Compute the information bottleneck at a fixed rate: the largest
            relevance ``I[Y : T | Z]`` of a representation T of X with
            complexity ``I[X : T | Z] <= rate`` :cite:`tishby2000information`.

            Parameters
            ----------
            dist : Distribution
                The distribution of interest.
            rate : float
                The maximum complexity, in bits.
            rvs : iter of iters, None
                The variables X and Y. If None, use ``dist.rvs``.
            crvs : iter, None
                The random variables to condition on.
            bound : int, None
                The bound on the size of T. If None, use ``|X||Z| + 1``.
            niter : int, None
                The number of basin hops to perform.
            maxiter : int, None
                The number of iterations of the optimization subroutine.
            rng : np.random.Generator, int, None
                Source of randomness for the optimizer.

            Returns
            -------
            relevance : float
                The maximal relevance.
            """
            ib = cls(dist=dist, rate=rate, rvs=rvs, crvs=crvs, bound=bound)
            ib.optimize(niter=niter, maxiter=maxiter, rng=rng)
            return max(float(ib.relevance(ib.construct_joint(ib._optima))), 0.0)

        return information_bottleneck_rate


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


information_bottleneck_rate = InformationBottleneckRate.functional()
