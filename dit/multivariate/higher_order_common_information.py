r"""
The higher-order common information :cite:`ostergaard2024higher`.

A reference variable :math:`X_i` is degraded through a chain of auxiliary
variables :math:`T_1 = X_i, T_2, \ldots, T_{n-1}`, processing the other variables
one at a time in some order :math:`\pi`. Processing :math:`X_a` is a
rate-constrained information bottleneck whose rate is what :math:`T_\ell` already
shares with :math:`X_a`,

.. math::

   T_{\ell+1} \in \arg\max \left\{ \I{S : X_a} : S - T_\ell - X_{0:n},\ \I{S : T_\ell} \leq \I{T_\ell : X_a} \right\},

and :math:`T_{\ell+1}` keeps only what is statistically relevant to :math:`X_a`
(it is target-minimal). The higher-order common information is the most the
final auxiliary can share with every variable,

.. math::

   \operatorname{HCI}[X_0 : \ldots : X_{n-1}] =
     \min_i \max_\pi \sup_{T_{n-1}} \min_j \I{T_{n-1} : X_j},

where the supremum runs over the chains of stage optimizers.
"""

from itertools import combinations

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..distribution import Distribution
from ..exceptions import OptimizationException
from ..helpers import normalize_rvs
from ..utils import unitful
from .common_informations import gk_common_information

__all__ = ("higher_order_common_information",)

# Information below this is zero.
_TOL = 1e-9

# Probabilities below this are dropped from the auxiliary variables.
_NEGLIGIBLE = 1e-12

# Conditional distributions within this distance (sup norm) are merged.
_MERGE_TOL = 1e-6

# Channels within this much of the stage optimum count as stage optimizers when
# breaking ties.
_STAGE_TOL = 1e-5


def _entropy(pmf):
    """
    The entropy, in bits, of a pmf array.
    """
    pmf = pmf[pmf > 0]
    return float(-(pmf * np.log2(pmf)).sum())


def _marginal(joint, axes):
    """
    The marginal of the joint pmf on ``axes``, in increasing axis order.
    """
    return joint.sum(axis=tuple(ax for ax in range(joint.ndim) if ax not in axes))


def _mutual_information(joint, a, b):
    """
    The mutual information between axes ``a`` and ``b`` of the joint pmf.
    """
    return _entropy(_marginal(joint, (a,))) + _entropy(_marginal(joint, (b,))) - _entropy(_marginal(joint, (a, b)))


def _relevances(state):
    """
    I[T : X_j] for each variable, where T is the last axis of ``state``.
    """
    t = state.ndim - 1
    return [_mutual_information(state, t, j) for j in range(t)]


def _joint_pmf(dist, rvs):
    """
    The joint pmf of the groups ``rvs``, without zero-probability symbols.
    """
    d = dist.coalesce(rvs)
    d.make_dense()
    joint = d.pmf.reshape([len(alphabet) for alphabet in d.alphabet])
    for axis in range(joint.ndim):
        joint = np.take(joint, np.flatnonzero(_marginal(joint, (axis,)) > 0), axis=axis)
    return joint


def _initial_state(joint, i):
    """
    The joint pmf of the variables and T_1 = X_i, which is the last axis.
    """
    k = joint.shape[i]
    copy = np.eye(k).reshape([k if ax == i else 1 for ax in range(joint.ndim)] + [k])
    return joint[..., np.newaxis] * copy


def _merge(state, target):
    """
    Replace the auxiliary variable (the last axis) by its minimal sufficient
    statistic for the variable ``target``: merge the values with the same
    conditional distribution of ``target``, and drop negligible values.
    """
    t = state.ndim - 1
    p_xt = _marginal(state, (target, t))
    p_t = p_xt.sum(axis=0)
    groups, posteriors = [], []
    for value in np.flatnonzero(p_t > _NEGLIGIBLE):
        posterior = p_xt[:, value] / p_t[value]
        for group, other in zip(groups, posteriors, strict=True):
            if np.abs(posterior - other).max() <= _MERGE_TOL:
                group.append(value)
                break
        else:
            groups.append([value])
            posteriors.append(posterior)
    merged = np.stack([state[..., group].sum(axis=-1) for group in groups], axis=-1)
    return merged / merged.sum()


def _as_distribution(state):
    """
    The pmf array as a Distribution over integer symbols.
    """
    support = np.argwhere(state > 0)
    pmf = state[tuple(support.T)]
    return Distribution([tuple(int(v) for v in index) for index in support], pmf / pmf.sum())


def _channel(opt, x):
    """
    The channel p(s | t) of the optimizer's auxiliary variable at ``x``, with
    rows indexed by the symbols of T, the optimizer's first rv group.
    """
    channel = next(opt._construct_channels(x.copy()))
    channel = channel.reshape(channel.shape[0], -1)
    out = np.empty_like(channel)
    for index, (t,) in opt._unqs[0].inverse.items():
        out[t] = channel[index]
    return out


def _vector(opt, channel):
    """
    The optimizer's optimization vector for a channel p(s | t) whose rows are
    indexed by the symbols of T; missing outputs get probability zero.
    """
    shape = opt._aux_vars[0].shape
    vec = np.zeros((shape[0], shape[-1]))
    for (t,), index in opt._unqs[0].mapping.items():
        vec[index, : channel.shape[1]] = channel[t]
    return vec.ravel()


class _TieBreak(BaseAuxVarOptimizer):
    """
    Among channels p(s | t) with I[T : S] <= budget and I[X_a : S] >= floor,
    maximize I[X_k : S] subject to I[X_j : S] >= I[X_k : S] for the other
    watched variables X_j, so that the best k maximizes min_j I[X_j : S].

    The rv groups are T, X_a, and then the watched variables.
    """

    _shotgun = 3
    _restarts = 2
    _feasibility_tol = 1e-6

    def __init__(self, dist, rvs, budget, floor, k, bound):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of the variables and T.
        rvs : list of lists
            T, X_a, and the watched variables.
        budget : float
            The largest allowed I[T : S].
        floor : float
            The smallest allowed I[X_a : S].
        k : int
            The index in ``rvs`` of the watched variable to maximize.
        bound : int
            The size of S.
        """
        super().__init__(dist, rvs=rvs)
        self._construct_auxvars([({0}, bound)])
        s = self._arvs
        infos = [self._mutual_information({v}, s) for v in range(len(rvs))]
        grads = [self._mutual_information_grad({v}, s) for v in range(len(rvs))]
        self._infos = infos
        self._k = k
        self._grad_k = grads[k]

        # Each slack is non-negative when its constraint holds.
        slacks = [
            (lambda p: budget - infos[0](p), lambda p: -grads[0](p)),
            (lambda p: infos[1](p) - floor, grads[1]),
        ]
        for j in range(2, len(rvs)):
            if j != k:
                slacks.append(
                    (lambda p, j=j: infos[j](p) - infos[k](p), lambda p, j=j: grads[j](p) - grads[k](p)),
                )
        self._slacks = [slack for slack, _ in slacks]

        def constraint(fun, grad):
            return {
                "type": "ineq",
                "fun": lambda x: fun(self.construct_joint(x)),
                "jac": lambda x: self._construct_joint_vjp(x, np.asarray(grad(self.construct_joint(x)))),
            }

        self.constraints = [constraint(fun, grad) for fun, grad in slacks]

    def _objective(self):
        """
        The negated information shared with the k-th watched variable.

        Returns
        -------
        objective : func
            The objective function.
        """
        info = self._infos[self._k]

        def objective(self, x):
            """
            Compute -I[X_k : S].

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            return -info(self.construct_joint(x))

        return objective

    def _objective_gradient(self):
        """
        Analytic gradient of the objective w.r.t. the joint.
        """
        return lambda pmf: -self._grad_k(pmf)

    def feasible(self, x):
        """
        Whether ``x`` satisfies every constraint.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        feasible : bool
            Whether the constraints hold.
        """
        pmf = self.construct_joint(x)
        return all(slack(pmf) >= -self._feasibility_tol for slack in self._slacks)

    def least_relevance(self, x):
        """
        The least information S shares with a watched variable.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        info : float
            min_j I[X_j : S] over the watched variables.
        """
        pmf = self.construct_joint(x)
        return min(info(pmf) for info in self._infos[2:])

    def search(self, seed, niter=None, maxiter=None, rng=None):
        """
        Search from ``seed`` and from random starts.

        Parameters
        ----------
        seed : np.ndarray
            An optimization vector to start from.
        niter : int, None
            The number of basin hops to perform.
        maxiter : int, None
            The number of iterations of the optimization subroutine.
        rng : np.random.Generator, None
            Source of randomness for the optimizer.

        Returns
        -------
        x : np.ndarray, None
            The best feasible optimization vector found, if any.
        """
        candidates = [seed]
        for restart in range(self._restarts):
            try:
                self.optimize(x0=seed if restart == 0 else None, niter=niter, maxiter=maxiter, rng=rng)
            except OptimizationException:
                continue
            candidates.append(self._optima)
        feasible = [x for x in candidates if self.feasible(x)]
        return max(feasible, key=self.least_relevance) if feasible else None


def _stage(state, a, watched, last, niter, maxiter, rng):
    """
    Process X_a: replace T_l, the last axis of ``state``, by a target-minimal
    stage optimizer T_{l+1}. Ties among stage optimizers go to the one sharing
    the most with the least related watched variable, which is exact at the
    last stage and an upper bound on what later stages can keep otherwise.

    Parameters
    ----------
    state : np.ndarray
        The joint pmf of the variables and T_l.
    a : int
        The variable to process.
    watched : list of int
        The variables other than X_a and the reference.
    last : bool
        Whether this is the last stage.
    niter, maxiter : int, None
        Passed to the optimizers.
    rng : np.random.Generator
        Source of randomness for the optimizers.

    Returns
    -------
    state : np.ndarray
        The joint pmf of the variables and T_{l+1}.
    """
    # Deferred: dit.rate_distortion imports dit.multivariate.
    from ..rate_distortion import InformationBottleneckRate

    t = state.ndim - 1
    budget = _mutual_information(state, t, a)
    if budget <= _TOL:
        # T_{l+1} is independent of T_l, and so of everything.
        return state.sum(axis=-1, keepdims=True)
    statistic = _merge(state, a)
    if _entropy(_marginal(statistic, (t,))) <= budget + _TOL:
        # What T_l knows about X_a is a function of X_a, and keeping exactly
        # that is the only target-minimal stage optimizer.
        return statistic

    dist = _as_distribution(state)
    ib = InformationBottleneckRate(dist, rate=budget, rvs=[[t], [a]])
    ib.optimize(niter=niter, maxiter=maxiter, rng=rng)
    best = _channel(ib, ib._optima)
    floor = ib.relevance(ib.construct_joint(ib._optima)) - _STAGE_TOL
    relevances = _relevances(np.tensordot(state, best, axes=(t, 0)))
    value = min(relevances[j] for j in watched)

    rvs = [[t], [a], *([j] for j in watched)]
    bound = state.shape[-1] + len(watched) + 1
    for k in range(2, len(rvs)):
        tie_break = _TieBreak(dist, rvs, budget, floor, k, bound)
        x = tie_break.search(_vector(tie_break, best), niter=niter, maxiter=maxiter, rng=rng)
        if x is not None and tie_break.least_relevance(x) > value:
            best, value = _channel(tie_break, x), tie_break.least_relevance(x)

    new = np.tensordot(state, best, axes=(t, 0))
    return new if last else _merge(new, a)


def _reference_value(state, i, stop, niter, maxiter, rng):
    """
    max_pi sup_T min_j I[T : X_j] for reference X_i, searching the orderings
    depth first so that orderings sharing a prefix share its stages. The search
    returns early once the value reaches ``stop``.

    Parameters
    ----------
    state : np.ndarray
        The joint pmf of the variables and T_1 = X_i.
    i : int
        The reference variable.
    stop : float
        Stop once the value reaches this.
    niter, maxiter : int, None
        Passed to the optimizers.
    rng : np.random.Generator
        Source of randomness for the optimizers.

    Returns
    -------
    value : float
        The value of reference X_i, or something at least ``stop``.
    """
    n = state.ndim - 1
    best = 0.0

    def search(state, remaining):
        nonlocal best
        if len(remaining) == 1:
            best = max(best, min(_relevances(state)))
            return
        for a in sorted(remaining):
            if best >= stop:
                return
            watched = [j for j in range(n) if j not in (i, a)]
            new = _stage(state, a, watched, len(remaining) == 2, niter, maxiter, rng)
            # Later stages only degrade T, so min_j I[T : X_j] bounds the branch.
            if min(_relevances(new)) > best + _TOL:
                search(new, remaining - {a})

    search(state, frozenset(range(n)) - {i})
    return best


@unitful
def higher_order_common_information(dist, rvs=None, niter=None, maxiter=None, rng=None):
    r"""
    Compute the higher-order common information :cite:`ostergaard2024higher`:

    .. math::

       \operatorname{HCI}[X_0 : \ldots : X_{n-1}] =
         \min_i \max_\pi \sup_{T_{n-1}} \min_j \I{T_{n-1} : X_j},

    where :math:`T_1 = X_i` and each :math:`T_{\ell+1}` maximizes
    :math:`\I{T_{\ell+1} : X_{\pi(\ell)}}` among target-minimal
    :math:`T_{\ell+1} - T_\ell - X_{0:n}` with
    :math:`\I{T_{\ell+1} : T_\ell} \leq \I{T_\ell : X_{\pi(\ell)}}`.

    For two variables it is the mutual information, and in general

    .. math::

       \K{X_0 : \ldots : X_{n-1}} \leq \operatorname{HCI}[X_0 : \ldots : X_{n-1}] \leq \min_{i \neq j} \I{X_i : X_j}.

    When the two bounds meet their common value is returned. Otherwise every
    ordering of every reference is searched, solving a rate-constrained
    information bottleneck at each stage, so the cost grows as :math:`n!`.
    Stages whose relevant part is a deterministic function of the processed
    variable are solved exactly; the rest are numerical. Ties among stage
    optimizers go to the one sharing the most with the least related other
    variable, which is exact for three variables and a heuristic for more.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The variables :math:`X_0, \ldots, X_{n-1}`. If None, use every
        variable of ``dist``.
    niter : int, None
        The number of basin hops per optimization.
    maxiter : int, None
        The number of iterations of the optimization subroutine.
    rng : np.random.Generator, int, None
        Source of randomness for the optimizers.

    Returns
    -------
    hci : float
        The higher-order common information.
    """
    rvs, _ = normalize_rvs(dist, rvs, None)
    joint = _joint_pmf(dist, rvs)
    n = joint.ndim
    if n == 1:
        return _entropy(joint)

    pairwise = np.full((n, n), np.inf)
    for i, j in combinations(range(n), 2):
        pairwise[i, j] = pairwise[j, i] = _mutual_information(joint, i, j)
    upper = float(pairwise.min())
    if n == 2 or upper <= _TOL:
        return max(upper, 0.0)
    lower = gk_common_information(dist, rvs)
    if lower >= upper - _TOL:
        return upper

    rng = np.random.default_rng(rng)
    value = upper
    # The reference with the weakest pairwise bound is the likeliest minimizer.
    for i in sorted(range(n), key=lambda i: pairwise[i].min()):
        value = min(value, _reference_value(_initial_state(joint, i), i, value, niter, maxiter, rng))
    return max(value, lower)
