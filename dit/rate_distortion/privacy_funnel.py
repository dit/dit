"""
The privacy funnel :cite:`makhdoumi2014information`: release a representation
T of public data X which reveals as little as possible about private data S,
subject to T being informative about X.
"""

from contextlib import suppress
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import linprog

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException
from .bottleneck_algorithms import _encoder_from_assignments, _greedy_privacy_funnel_path
from .information_bottleneck import partition_encoders

__all__ = (
    "PrivacyFunnel",
    "PrivacyFunnelRate",
    "admits_perfect_privacy",
    "perfect_privacy_utility",
    "privacy_funnel",
    "privacy_funnel_rate",
)

_TOL = 1e-10


def _joint_pmf(dist, rvs):
    """
    The joint pmf p(s, x) with axis 0 indexing S and axis 1 indexing X.
    """
    if rvs is None:
        rvs = dist.rvs
    if len(rvs) != 2:
        msg = "The privacy funnel is only defined for two variables."
        raise ditException(msg)
    joint = dist.coalesce(rvs)
    joint.make_dense()
    return joint.pmf.reshape([len(a) for a in joint.alphabet])


def _entropy(pmf):
    pmf = np.asarray(pmf, dtype=float)
    pmf = pmf[pmf > 0]
    return float(-(pmf * np.log2(pmf)).sum())


def _privacy_matrix(p_sx):
    """
    The columns p(S | x) over the support of X, the support mask, and p(S).
    """
    p_x = p_sx.sum(axis=0)
    support = p_x > _TOL
    return p_sx[:, support] / p_x[support], support, p_sx.sum(axis=1)


def _perfect_privacy_vertices(p_sx, max_candidates):
    """
    The extreme points of the polytope {q in the simplex over X : P_{S|X} q = p_S}
    of distributions on X under which S is distributed as p_S. They are the basic
    feasible solutions of the linear system, one per nonsingular column basis.
    """
    a, _, p_s = _privacy_matrix(p_sx)
    n_x = a.shape[1]
    rank = np.linalg.matrix_rank(a, tol=1e-9)
    if comb(n_x, rank) > max_candidates:
        msg = f"Too many candidate bases ({comb(n_x, rank)}) to enumerate the perfect privacy polytope."
        raise ditException(msg)

    vertices = []
    for basis in combinations(range(n_x), rank):
        sub = a[:, basis]
        if np.linalg.matrix_rank(sub, tol=1e-9) < rank:
            continue
        q_b = np.linalg.lstsq(sub, p_s, rcond=None)[0]
        if q_b.min() < -1e-9 or np.abs(sub @ q_b - p_s).max() > 1e-9:
            continue
        q = np.zeros(n_x)
        q[list(basis)] = np.clip(q_b, 0, None)
        q /= q.sum()
        if not any(np.allclose(q, v, atol=1e-9) for v in vertices):
            vertices.append(q)
    return np.array(vertices)


def _perfect_privacy_lp(p_sx, max_candidates):
    """
    Maximize I(X : T) subject to I(S : T) = 0 :cite:`rassouli2021perfect`.

    Any such T decomposes p_X into a mixture of points of the perfect privacy
    polytope, and I(X : T) = H(X) - sum_t p(t) H(X | T = t). Entropy is concave,
    so the optimal mixture uses only vertices, and the weights solve a linear
    program.

    Returns
    -------
    g0 : float
        The maximal utility at zero leakage.
    p_t_given_x : np.ndarray
        An optimal mapping, of shape ``(|X|, |T|)``.
    """
    p_x = p_sx.sum(axis=0)
    _, support, _ = _privacy_matrix(p_sx)
    vertices = _perfect_privacy_vertices(p_sx, max_candidates)
    costs = np.array([_entropy(v) for v in vertices])
    res = linprog(costs, A_eq=vertices.T, b_eq=p_x[support], bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover
        msg = f"Perfect privacy linear program failed: {res.message}"
        raise ditException(msg)

    weights = np.where(res.x > _TOL, res.x, 0.0)
    used = np.flatnonzero(weights)
    p_t_given_x = np.zeros((len(p_x), len(used)))
    p_t_given_x[support] = (vertices[used] * weights[used, np.newaxis]).T / p_x[support, np.newaxis]
    p_t_given_x[~support, 0] = 1.0
    p_t_given_x /= p_t_given_x.sum(axis=1, keepdims=True)

    g0 = max(_entropy(p_x) - float(costs @ res.x), 0.0)
    return g0, p_t_given_x


def admits_perfect_privacy(dist, rvs=None):
    """
    Determine whether X can be released with non-zero utility and zero leakage
    about S.

    This is the case if and only if the vectors p(S | x), over the support of X,
    are linearly dependent :cite:`calmon2015fundamental`; in particular,
    whenever ``|X| > |S|``.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, None
        The private variables S and the public variables X. If None, use
        ``dist.rvs``.

    Returns
    -------
    perfect : bool
        Whether perfect privacy is possible.
    """
    a, _, _ = _privacy_matrix(_joint_pmf(dist, rvs))
    return bool(np.linalg.matrix_rank(a, tol=1e-9) < a.shape[1])


def perfect_privacy_utility(dist, rvs=None, return_mechanism=False, max_candidates=100_000):
    """
    Compute the maximal utility at perfect privacy,

    .. math::

        g_0 = \\max_{p(t | x) : I(S : T) = 0} I(X : T),

    exactly, by enumerating the vertices of the perfect privacy polytope and
    solving a linear program :cite:`rassouli2021perfect`.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, None
        The private variables S and the public variables X. If None, use
        ``dist.rvs``.
    return_mechanism : bool
        If True, also return an optimal mapping p(t | x).
    max_candidates : int
        The maximum number of column bases to enumerate.

    Returns
    -------
    g0 : float
        The maximal utility, in bits, of a release leaking nothing about S.
    p_t_given_x : np.ndarray
        An optimal mapping of shape ``(|X|, |T|)``, if ``return_mechanism``.

    Raises
    ------
    ditException
        If the polytope has too many candidate bases.
    """
    g0, p_t_given_x = _perfect_privacy_lp(_joint_pmf(dist, rvs), max_candidates)
    if return_mechanism:
        return g0, p_t_given_x
    return g0


class PrivacyFunnel(BaseAuxVarOptimizer):
    """
    The privacy funnel Lagrangian :cite:`makhdoumi2014information`,

    .. math::

        \\min_{p(t | x, z)} I[S : T | Z] - \\beta I[X : T | Z],

    over mappings T of X, so that S - XZ - T. Since ``I[S : T | Z] <= I[X : T |
    Z]``, every ``beta >= 1`` is minimized by releasing ``T = X``, so the whole
    curve is traced by ``beta`` in [0, 1]. The objective is a difference of
    convex functions of the mapping, so it is non-convex
    :cite:`asoodeh2020bottleneck`.
    """

    _shotgun = 10
    _max_partition_seeds = 2000

    # The Lagrangian is minimized on the upper concave envelope of a function on
    # the simplex over X, which needs at most |X| points (Caratheodory); a rate
    # constraint adds one more.
    _extra_symbols = 0

    def __init__(self, dist, beta, rvs=None, crvs=None, bound=None):
        """
        Initialize the privacy funnel.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        beta : float
            The non-negative price of utility.
        rvs : iterable of iterables, None
            The private variables S and the public variables X. If None, use
            ``dist.rvs``.
        crvs : iterable, None
            The random variables to condition on.
        bound : int, None
            The bound on the size of T. If None, use ``|X|``.
        """
        if rvs is None:
            rvs = dist.rvs

        if len(rvs) != 2:
            msg = "The privacy funnel is only defined for two variables."
            raise ditException(msg)

        super().__init__(dist=dist, rvs=rvs, crvs=crvs)
        if not beta >= 0.0:
            msg = "beta must be non-negative."
            raise ditException(msg)
        self._beta = beta

        self._s = {0}
        self._x = {1}
        self._z = {2}
        self._t = {3}

        tbound = self._shape[1] + self._extra_symbols
        bound = min(bound, tbound) if bound is not None else tbound

        self._construct_auxvars([(self._x | self._z, bound)])

        self.leakage = self._conditional_mutual_information(self._s, self._t, self._z)
        self.utility = self._conditional_mutual_information(self._x, self._t, self._z)

        self._default_hops *= 3

    def _objective(self):
        """
        Compute the appropriate objective.

        Returns
        -------
        objective : func
            The objective function.
        """

        def objective(self, x):
            """
            Compute I[S : T | Z] - \\beta I[X : T | Z].

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
            return self.leakage(pmf) - self._beta * self.utility(pmf)

        return objective

    def _objective_gradient(self):
        """
        Analytic gradient of the objective w.r.t. the joint.
        """
        leakage_g = self._conditional_mutual_information_grad(self._s, self._t, self._z)
        utility_g = self._conditional_mutual_information_grad(self._x, self._t, self._z)

        def grad(pmf):
            return leakage_g(pmf) - self._beta * utility_g(pmf)

        return grad

    def construct_copy_initial(self):
        """
        The identity mapping T = X, within each value of Z.

        Returns
        -------
        x : np.ndarray
            An optimization vector.
        """
        n_x, n_z, bound = self._aux_vars[0].shape
        enc = np.zeros((n_x, n_z, bound))
        enc[np.arange(n_x), :, np.minimum(np.arange(n_x), bound - 1)] = 1
        return enc.ravel()

    def _vector_from_mapping(self, p_t_given_x):
        """
        An optimization vector for an unconditional mapping p(t | x), or None if
        it uses more symbols than the bound allows.
        """
        n_x, _, bound = self._aux_vars[0].shape
        if p_t_given_x.shape[1] > bound:
            return None
        enc = np.zeros((n_x, 1, bound))
        enc[:, 0, : p_t_given_x.shape[1]] = p_t_given_x
        return enc.ravel()

    def _structured_mappings(self):
        """
        Candidate mappings from the greedy merge path
        :cite:`makhdoumi2014information` and the perfect privacy linear program
        :cite:`rassouli2021perfect`, when Z is trivial.
        """
        if self._shape[2] != 1:
            return []
        p_sx = self._pmf[:, :, 0]
        path, _ = _greedy_privacy_funnel_path(p_sx, rate=getattr(self, "_rate", 0.0))
        mappings = [_encoder_from_assignments(assignments) for assignments in path]
        with suppress(ditException):
            mappings.append(_perfect_privacy_lp(p_sx, max_candidates=10_000)[1])
        return [v for v in map(self._vector_from_mapping, mappings) if v is not None]

    def _seed_initials(self):
        """
        The constant and identity mappings, plus the greedy and perfect privacy
        mappings when Z is trivial.

        Returns
        -------
        seeds : list of np.ndarray
            Optimization vectors.
        """
        return [self.construct_constant_initial(), self.construct_copy_initial(), *self._structured_mappings()]

    def _candidates(self):
        """
        The seeds plus every deterministic mapping, when there are few enough.
        """
        partitions = [enc.ravel() for enc in partition_encoders(self._aux_vars[0].shape, self._max_partition_seeds)]
        return [*self._seed_initials(), *partitions]

    def optimize(self, *args, **kwargs):
        """
        Optimize, then keep the best of the result and the candidate mappings,
        since local search started at a seed can drift away from it.

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
        self._optima = min([self._optima, *self._candidates()], key=self.objective).copy()
        return result

    @classmethod
    def functional(cls):
        """
        Return a function which computes the result of this optimization.

        Returns
        -------
        privacy_funnel : func
            The function which performs this optimization.
        """

        def privacy_funnel(dist, beta, rvs=None, crvs=None, bound=None):
            """
            Compute a point on the privacy funnel by minimizing
            ``I[S : T | Z] - beta I[X : T | Z]`` :cite:`makhdoumi2014information`.

            Parameters
            ----------
            dist : Distribution
                The distribution of interest.
            beta : float
                The non-negative price of utility.
            rvs : iterable of iterables, None
                The private variables S and the public variables X. If None,
                use ``dist.rvs``.
            crvs : iterable, None
                The random variables to condition on.
            bound : int, None
                The bound on the size of T. If None, use ``|X|``.

            Returns
            -------
            leakage : float
                I[S : T | Z] at the optimum.
            utility : float
                I[X : T | Z] at the optimum.
            """
            pf = cls(dist=dist, beta=beta, rvs=rvs, crvs=crvs, bound=bound)
            pf.optimize()
            pmf = pf.construct_joint(pf._optima)
            return pf.leakage(pmf), pf.utility(pmf)

        return privacy_funnel


class PrivacyFunnelRate(PrivacyFunnel):
    """
    The privacy funnel :cite:`makhdoumi2014information`,

    .. math::

        \\mathrm{PF}(R) = \\min_{p(t | x, z) : I[X : T | Z] \\geq R} I[S : T | Z].

    The size of T is bounded by ``|X| + 1``; unlike the information bottleneck,
    ``|X|`` does not suffice in general :cite:`asoodeh2020bottleneck`.
    """

    _extra_symbols = 1

    # SLSQP and polishing leave the optimum on the active constraint up to about
    # this much slack.
    _feasibility_tol = 1e-6

    # Independent basin-hopping searches; a single one occasionally ends in a
    # poor local minimum.
    _restarts = 3

    def __init__(self, dist, rate, rvs=None, crvs=None, bound=None):
        """
        Initialize the privacy funnel.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rate : float
            The minimum utility R, in bits.
        rvs : iterable of iterables, None
            The private variables S and the public variables X. If None, use
            ``dist.rvs``.
        crvs : iterable, None
            The random variables to condition on.
        bound : int, None
            The bound on the size of T. If None, use ``|X| + 1``.
        """
        self._rate = rate
        super().__init__(dist=dist, beta=0.0, rvs=rvs, crvs=crvs, bound=bound)

        p_xz = self._pmf.sum(axis=0)
        max_rate = _entropy(p_xz) - _entropy(p_xz.sum(axis=0))
        if not 0.0 <= rate <= max_rate + 1e-9:
            msg = "rate must lie in [0, H[X | Z]]."
            raise ditException(msg)

        utility_g = self._conditional_mutual_information_grad(self._x, self._t, self._z)
        self.constraints += [
            {
                "type": "ineq",
                "fun": self._constraint_utility,
                "jac": lambda x: self._construct_joint_vjp(x, np.asarray(utility_g(self.construct_joint(x)))),
            },
        ]

    def _constraint_utility(self, x):
        """
        Non-negative when the utility meets the rate.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        slack : float
            ``I[X : T | Z] - R``.
        """
        return self.utility(self.construct_joint(x)) - self._rate

    def optimize(self, *args, **kwargs):
        """
        Optimize from the identity mapping, which is always feasible, a few
        times, then keep the best feasible candidate.

        Parameters
        ----------
        args, kwargs
            Passed to :meth:`BaseAuxVarOptimizer.optimize`.

        Returns
        -------
        result : OptimizeResult
            The result of the search.
        """
        if not args and kwargs.get("x0") is None:
            kwargs["x0"] = self.construct_copy_initial()
        optima = []
        for _ in range(self._restarts):
            result = BaseAuxVarOptimizer.optimize(self, *args, **kwargs)
            optima.append(self._optima)
        feasible = [x for x in [*optima, *self._candidates()] if self._constraint_utility(x) >= -self._feasibility_tol]
        self._optima = min(feasible, key=self.objective).copy()
        return result

    @classmethod
    def functional(cls):
        """
        Return a function which computes the result of this optimization.

        Returns
        -------
        privacy_funnel_rate : func
            The function which performs this optimization.
        """

        def privacy_funnel_rate(dist, rate, rvs=None, crvs=None, bound=None):
            """
            Compute the privacy funnel, the least leakage ``I[S : T | Z]`` of a
            release T of X with utility ``I[X : T | Z] >= rate``
            :cite:`makhdoumi2014information`.

            Parameters
            ----------
            dist : Distribution
                The distribution of interest.
            rate : float
                The minimum utility, in bits.
            rvs : iterable of iterables, None
                The private variables S and the public variables X. If None,
                use ``dist.rvs``.
            crvs : iterable, None
                The random variables to condition on.
            bound : int, None
                The bound on the size of T. If None, use ``|X| + 1``.

            Returns
            -------
            pf : float
                The minimal leakage.
            """
            pf = cls(dist=dist, rate=rate, rvs=rvs, crvs=crvs, bound=bound)
            pf.optimize()
            return max(float(pf.leakage(pf.construct_joint(pf._optima))), 0.0)

        return privacy_funnel_rate


privacy_funnel = PrivacyFunnel.functional()
privacy_funnel_rate = PrivacyFunnelRate.functional()
