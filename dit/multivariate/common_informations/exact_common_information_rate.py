r"""
The exact common information rate.

Kumar, Li, and El Gamal :cite:`kumar2014exact` define the exact common
information rate of a pair :math:`X, Y` as the least rate of common randomness,
compressed with variable-length codes, from which two parties can separately
generate :math:`X^n` and :math:`Y^n` distributed exactly as :math:`n` i.i.d.
draws of :math:`p(x, y)`. It is

.. math::

   \bar{G}[X : Y] = \lim_{n \to \infty} \frac{1}{n} \G{X^n : Y^n}
                  = \inf_n \frac{1}{n} \G{X^n : Y^n},

where :math:`\G{X : Y}` is the one-shot :func:`exact_common_information`, which
is subadditive. Yu and Tan :cite:`yu2020on,yu2020exact` bound the rate with the
:func:`~dit.divergences.maximal_cross_entropy` :math:`\mathbb{H}`:

.. math::

   \Gamma^{(o)}[X : Y] \leq \bar{G}[X : Y] \leq \Gamma[X : Y]
     = \min_{X - W - Y} -\H{X, Y | W} + \sum_w p(w) \mathbb{H}(p_{X|w}, p_{Y|w} \| p_{XY}),

where the lower bound :math:`\Gamma^{(o)}` replaces the pairing of
:math:`p_{X|w}` with :math:`p_{Y|w}` by the cheapest coupling of :math:`p_W`
with itself,

.. math::

   \Gamma^{(o)}[X : Y] = \min_{X - W - Y} -\H{X, Y | W}
     + \min_{q \in \mathcal{C}(p_W, p_W)} \sum_{w, w'} q(w, w') \mathbb{H}(p_{X|w}, p_{Y|w'} \| p_{XY}).

The upper bound is achievable by any feasible :math:`W`, satisfies
:math:`\C{X : Y} \leq \Gamma[X : Y] \leq \G{X : Y}`, and is tight for the doubly
symmetric binary source. Applied to :math:`n` i.i.d. copies it gives upper bounds
:math:`\Gamma[X^n : Y^n] / n` that decrease to :math:`\bar{G}` as :math:`n` grows.
"""

import threading

import numpy as np
from scipy.optimize import linprog, minimize

from ...distribution import Distribution
from ...divergences.maximal_cross_entropy import _max_coupling, _transport
from ...exceptions import OptimizationException, ditException
from ...helpers import normalize_rvs
from ...utils.units import _rescale
from ..dual_total_correlation import dual_total_correlation
from .base_markov_optimizer import MarkovVarOptimizer
from .exact_common_information import ExactCommonInformation
from .wyner_common_information import WynerCommonInformation

__all__ = (
    "exact_common_information_rate",
    "exact_common_information_rate_bounds",
)

# Auxiliary states lighter than this are unused.
_MASS_TOL = 1e-12
# Auxiliary states lighter than this are paired only with themselves in the lower
# bound's coupling of p(w) with itself, keeping its linear program well scaled.
_DUST_TOL = 1e-9
# Conditional probabilities at most this are treated as zero when deciding which
# pairs of states may be coupled in the lower bound.
_SUPPORT_TOL = 1e-9
# Largest deviation of a refined decomposition's joint from the distribution.
_MATCH_TOL = 1e-9
# Conditional probabilities below this are set to zero before mixing states.
_CLEAN_TOL = 1e-7
# Largest deviation of a mixture's joint from the distribution accepted for the
# lower bound, which may err low.
_LOOSE_TOL = 1e-6
# Tight tolerances for the mixtures' linear programs.
_LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def _cost_matrix(pxy):
    """
    The cost ``-log2 p(x, y)`` of the maximal cross entropy. Cells where
    ``p(x, y)`` vanishes cost one bit more than any real cell; a feasible
    decomposition never couples mass into them, so the penalty only shapes the
    path of the optimizer.
    """
    live = pxy > 0
    cost = np.zeros_like(pxy, dtype=float)
    cost[live] = -np.log2(pxy[live])
    cost[~live] = cost[live].max() + 1.0
    return cost


def _conditionals(joint):
    """The used states of `joint` (shape ``(|X|, |Y|, |W|)``), their masses, and p(x|w), p(y|w)."""
    mass = joint.sum(axis=(0, 1))
    live = np.flatnonzero(mass > _MASS_TOL)
    px_w = joint.sum(axis=1)[:, live].T / mass[live, np.newaxis]
    py_w = joint.sum(axis=0)[:, live].T / mass[live, np.newaxis]
    return live, mass[live], px_w, py_w


def _inner_sum_rate(joint, cost, zero=None):
    r"""
    The coupling term of the upper bound,
    :math:`\sum_w p(w) \mathbb{H}(p_{X|w}, p_{Y|w} \| p_{XY})`, and its gradient.

    Parameters
    ----------
    joint : np.ndarray, shape (|X|, |Y|, |W|)
        The joint distribution :math:`p(x, y, w)`.
    cost : np.ndarray, shape (|X|, |Y|)
        The costs from :func:`_cost_matrix`.
    zero : np.ndarray, None
        Unused; accepted for symmetry with :func:`_outer_sum_rate`.

    Returns
    -------
    value : float
        The coupling term.
    grad : np.ndarray, shape (|X|, |Y|, |W|)
        Its (super)gradient with respect to `joint`.
    """
    live, mass, px_w, py_w = _conditionals(joint)
    # Mass added to an unused state is a point mass at (x, y), whose only coupling costs cost[x, y].
    grad = np.repeat(cost[:, :, np.newaxis], joint.shape[2], axis=2)
    if live.size == 0:
        return 0.0, grad
    values, f, g = _max_coupling(px_w, py_w, cost)
    grad[:, :, live] = (f[:, :, np.newaxis] + g[:, np.newaxis, :]).transpose(1, 2, 0)
    return float(mass @ values), grad


def _diagonal_is_optimal(pair_costs, forbidden):
    r"""
    Whether pairing every state with itself is an optimal coupling of
    :math:`p_W` with itself. Starting from the diagonal, mass can only be moved
    around a cycle :math:`w_1 \to w_2 \to \cdots \to w_1`, changing the cost by
    :math:`\sum_i M_{w_i w_{i+1}} - M_{w_i w_i}`; the diagonal is optimal exactly
    when no such cycle is negative.
    """
    lengths = np.where(forbidden, np.inf, pair_costs - np.diag(pair_costs)[:, np.newaxis])
    np.fill_diagonal(lengths, 0.0)
    for i in range(len(lengths)):
        lengths = np.minimum(lengths, lengths[:, i : i + 1] + lengths[i : i + 1, :])
    return np.diag(lengths).min() >= -1e-12


def _outer_sum_rate(joint, cost, zero):
    r"""
    The coupling term of the lower bound,
    :math:`\min_{q \in \mathcal{C}(p_W, p_W)} \sum_{w, w'} q(w, w') \mathbb{H}(p_{X|w}, p_{Y|w'} \| p_{XY})`,
    and its gradient.

    A pair of states whose coupling can place mass where :math:`p(x, y) = 0` has
    infinite maximal cross entropy, so it is excluded from the coupling of
    :math:`p_W` with itself; pairing each state with itself is always allowed.

    Parameters
    ----------
    joint : np.ndarray, shape (|X|, |Y|, |W|)
        The joint distribution :math:`p(x, y, w)`.
    cost : np.ndarray, shape (|X|, |Y|)
        The costs from :func:`_cost_matrix`.
    zero : np.ndarray, shape (|X|, |Y|)
        Where :math:`p(x, y) = 0`.

    Returns
    -------
    value : float
        The coupling term.
    grad : np.ndarray, shape (|X|, |Y|, |W|)
        Its gradient with respect to `joint`.
    """
    live, p, px_w, py_w = _conditionals(joint)
    grad = np.repeat(cost[:, :, np.newaxis], joint.shape[2], axis=2)
    k = live.size
    if k == 0:
        return 0.0, grad
    m, n = cost.shape

    # Maximal cross entropy of every ordered pair (w, v), pair (w, v) at index w * k + v.
    values, f, g = _max_coupling(np.repeat(px_w, k, axis=0), np.tile(py_w, (k, 1)), cost)
    pair_costs = values.reshape(k, k)
    infinite = ((px_w > _SUPPORT_TOL).astype(float) @ zero @ (py_w > _SUPPORT_TOL).T.astype(float)) > 0
    np.fill_diagonal(infinite, False)
    if _diagonal_is_optimal(pair_costs, infinite):
        coupling = np.diag(p)
        prices = np.diag(pair_costs)  # complementary slackness on the diagonal
    else:
        # States too light for the solver's tolerances stay paired with themselves.
        heavy = np.flatnonzero(p > _DUST_TOL)
        plans, u, v = _transport(
            p[np.newaxis, heavy],
            p[np.newaxis, heavy],
            pair_costs[np.ix_(heavy, heavy)][np.newaxis],
            maximize=False,
            forbidden=infinite[np.ix_(heavy, heavy)],
        )
        coupling = np.diag(p)
        coupling[np.ix_(heavy, heavy)] = plans[0]
        prices = np.diag(pair_costs).copy()
        prices[heavy] = u[0] + v[0]

    # Envelope theorem: differentiate each pair cost through the conditionals it depends on.
    f = f.reshape(k, k, m)
    g = g.reshape(k, k, n)
    f = f - (px_w[:, np.newaxis, :] * f).sum(axis=2, keepdims=True)
    g = g - (py_w[np.newaxis, :, :] * g).sum(axis=2, keepdims=True)
    gx = np.einsum("wv,wvx->wx", coupling, f) / p[:, np.newaxis]
    gy = np.einsum("wv,wvy->vy", coupling, g) / p[:, np.newaxis]
    block = gx[:, :, np.newaxis] + gy[:, np.newaxis, :] + prices[:, np.newaxis, np.newaxis]
    grad[:, :, live] = block.transpose(1, 2, 0)
    return float((coupling * pair_costs).sum()), grad


class ExactCommonInformationRate(MarkovVarOptimizer):
    """
    Minimize the Yu-Tan sum rate over auxiliary variables W that render two
    groups of variables conditionally independent: the upper bound on the exact
    common information rate, or with ``outer=True`` the lower bound. With an
    ``input_weight`` :math:`\\lambda`, minimize
    :math:`(1 - \\lambda) \\Gamma_W + \\lambda I[X : W]` instead, whose minima trace
    the boundary of the exact channel synthesis region.
    """

    name = "exact rate"
    description = "min -H[X, Y | W] + sum_w p(w) MCE(p(X|w), p(Y|w) || p(X, Y)) over W rendering X, Y independent"

    _shotgun = 5

    def __init__(self, dist, rvs=None, crvs=None, bound=None, outer=False, input_weight=0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rvs : list, None
            Two lists of indexes, the variables of X and of Y.
        crvs : list, None
            Must be empty; the bounds are not defined conditionally.
        bound : int, None
            Place an artificial bound on the size of W.
        outer : bool
            Minimize the lower bound instead of the upper bound.
        input_weight : float
            The weight :math:`\\lambda \\in [0, 1]` of :math:`I[X : W]` in the
            objective. Must be 0 when `outer` is True.
        """
        rvs, crvs = normalize_rvs(dist, rvs, crvs)
        if len(rvs) != 2:
            msg = f"The exact common information rate bounds are defined for two groups of variables, not {len(rvs)}."
            raise ditException(msg)
        if crvs:
            msg = "The exact common information rate bounds are not defined conditionally."
            raise ditException(msg)
        self._outer = outer
        self._input_weight = input_weight
        self._seeds = []
        self._objective_bound = dual_total_correlation(dist, rvs, crvs) if input_weight == 0 else None
        super().__init__(dist, rvs=rvs, crvs=crvs, bound=bound)

        pxy = self._pmf_to_match[:, :, 0]
        self._cost = _cost_matrix(pxy)
        self._zero = (pxy <= 0).astype(float)
        self._sum_rate = _outer_sum_rate if outer else _inner_sum_rate

    def compute_bound(self):
        """
        An upper bound on the cardinality of the auxiliary random variable,
        :math:`|X||Y| + 1`, from the Caratheodory-Fenchel theorem.

        Returns
        -------
        bound : int
            The upper bound on the alphabet size of the auxiliary variable.
        """
        return int(np.prod(self._shape)) + 1

    def _coupling_term(self, pmf):
        """The coupling term and its gradient, memoized per thread on the joint."""
        store = self.__dict__.get("_coupling_cache")
        if store is None:
            store = self._coupling_cache = threading.local()
        cache = getattr(store, "cache", None)
        if cache is None:
            cache = store.cache = {}
        key = pmf.tobytes()
        if key not in cache:
            if len(cache) >= 16:
                cache.pop(next(iter(cache)))
            cache[key] = self._sum_rate(np.asarray(pmf, dtype=float)[:, :, 0, :], self._cost, self._zero)
        return cache[key]

    def _objective(self):
        """
        The sum rate, :math:`-H[X, Y | W]` plus the coupling term, mixed with
        :math:`I[X : W]` by the input weight.

        Returns
        -------
        obj : func
            The objective function.
        """
        conditional_entropy = self._entropy(self._rvs, self._W | self._crvs)
        input_information = self._conditional_mutual_information({0}, self._W, self._crvs)
        weight = self._input_weight

        def objective(self, x):
            """
            Compute the objective.

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
            sum_rate = self._coupling_term(pmf)[0] - conditional_entropy(pmf)
            if weight:
                return (1 - weight) * sum_rate + weight * input_information(pmf)
            return sum_rate

        return objective

    def _objective_gradient(self):
        """Gradient of the objective with respect to the joint."""
        conditional_entropy_grad = self._entropy_grad(self._rvs, self._W | self._crvs)
        input_information_grad = self._conditional_mutual_information_grad({0}, self._W, self._crvs)
        weight = self._input_weight

        def grad(pmf):
            g = -conditional_entropy_grad(pmf)
            g[:, :, 0, :] += self._coupling_term(pmf)[1]
            if weight:
                g = (1 - weight) * g + weight * input_information_grad(pmf)
            return g

        return grad

    def _seed_initials(self):
        """
        The seed decompositions, minimized before the random starts.

        Returns
        -------
        seeds : list of np.ndarray
            The seed optimization vectors.
        """
        return list(self._seeds)


###############################################################################
# Decompositions of integer-coded pairs.
#
# Every optimizer below runs on a two-variable distribution whose values are the
# row and column indexes ("codes") of a probability matrix p(x, y). A
# decomposition is a pair of arrays ``(p(w|x), p(y|w))`` indexed by codes, which
# makes the decomposition of n i.i.d. copies a Kronecker product.


def _pair_matrix(dist, rvs):
    """The probability matrix of the pair of variable groups `rvs` of `dist`."""
    d = dist.copy(base="linear").coalesce(rvs)
    d.make_sparse()
    xs = sorted({o[0] for o in d.outcomes}, key=repr)
    ys = sorted({o[1] for o in d.outcomes}, key=repr)
    pxy = np.zeros((len(xs), len(ys)))
    for o, p in zip(d.outcomes, d.pmf, strict=True):
        pxy[xs.index(o[0]), ys.index(o[1])] += p
    return pxy


def _pair_distribution(pxy):
    """A two-variable distribution over the codes ``(i, j)`` of the support of `pxy`."""
    cells = np.argwhere(pxy > 0)
    return Distribution([(int(i), int(j)) for i, j in cells], [pxy[i, j] for i, j in cells])


def _codes(opt, k):
    """For each index of the `k`-th variable in `opt`'s internal order, its code."""
    inverse = opt._unqs[k].inverse
    return np.array([inverse[i][0] for i in range(len(inverse))])


def _decomposition(opt, x):
    """The decomposition ``(p(w|x), p(y|w))`` of the used states, indexed by codes."""
    channels = list(opt._construct_channels(np.array(x, dtype=float)))
    p_w_x = channels[0][:, 0, :]
    p_y_w = channels[1][0]
    pxy = opt._pmf_to_match[:, :, 0]
    used = (pxy.sum(axis=1) @ p_w_x) > _MASS_TOL
    p_w_x, p_y_w = p_w_x[:, used], p_y_w[used]
    out_w_x = np.zeros_like(p_w_x)
    out_w_x[_codes(opt, 0)] = p_w_x
    out_y_w = np.zeros_like(p_y_w)
    out_y_w[:, _codes(opt, 1)] = p_y_w
    return out_w_x, out_y_w


def _vector(opt, p_w_x, p_y_w):
    """
    The optimization vector of `opt` for a decomposition indexed by codes, or
    None if it uses more states than `opt` allows. Unused states get no mass.
    """
    bound = opt._aux_bounds[0]
    states = p_w_x.shape[1]
    if states > bound:
        return None
    channel_w = np.zeros((p_w_x.shape[0], 1, bound))
    channel_w[:, 0, :states] = p_w_x[_codes(opt, 0)]
    channel_y = np.full((1, bound, p_y_w.shape[1]), 1 / p_y_w.shape[1])
    channel_y[0, :states] = p_y_w[:, _codes(opt, 1)]
    return np.concatenate([channel_w.ravel(), channel_y.ravel()])


def _refine(opt, x, maxiter):
    """
    Re-minimize from `x` with the joint matched cell by cell. The optimizers'
    single squared residual flattens out near feasibility, so their optima match
    the distribution only loosely, and a loosely matching auxiliary variable can
    undercut the true minimum. Matching every cell pins the joint down.

    The sum rate has kinks wherever an optimal coupling changes, which can
    strand the refinement, so it is also run on the smooth surrogate
    :math:`I[X, Y : W]`, a lower bound on the sum rate that shares its minimizer
    whenever each state makes one of the two conditionals deterministic.

    Returns
    -------
    refined : list of np.ndarray
        The refined optimization vectors that match the distribution to within
        ``_MATCH_TOL``.
    """
    target = opt._pmf_to_match[:, :, 0].ravel()[:-1]  # the last cell is implied by the others

    def match(y):
        return opt.construct_joint(y).sum(axis=-1)[:, :, 0].ravel()[:-1] - target

    mutual_information = opt._conditional_mutual_information(opt._rvs, opt._W, opt._crvs)
    mutual_information_grad = opt._conditional_mutual_information_grad(opt._rvs, opt._W, opt._crvs)

    def surrogate(y):
        return mutual_information(opt.construct_joint(y))

    def surrogate_jac(y):
        return opt._construct_joint_vjp(y, mutual_information_grad(opt.construct_joint(y)))

    refined = []
    for fun, jac in [(opt.objective, opt._jacobian), (surrogate, surrogate_jac)]:
        res = minimize(
            fun,
            x,
            jac=jac,
            method="SLSQP",
            bounds=[(0, 1)] * x.size,
            constraints=[{"type": "eq", "fun": match}],
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        if np.abs(match(res.x)).max() <= _MATCH_TOL:
            refined.append(res.x)
    return refined


def _entropies(rows):
    """The entropy of each row of `rows`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(rows * np.log2(rows), axis=1)


def _pairs(pxy, p_w_x, p_y_w):
    """
    The conditional pairs ``(p(x|w), p(y|w))`` of a decomposition indexed by
    codes, with probabilities below ``_CLEAN_TOL`` set to zero.
    """
    joint_wx = pxy.sum(axis=1)[:, np.newaxis] * p_w_x
    px_w = (joint_wx / joint_wx.sum(axis=0)).T
    py_w = np.array(p_y_w, dtype=float)
    for rows in (px_w, py_w):
        rows[rows < _CLEAN_TOL] = 0.0
        rows /= rows.sum(axis=1, keepdims=True)
    return px_w, py_w


def _trivial_pairs(pxy):
    """
    The conditional pairs of the decompositions every distribution has:
    :math:`W = X`, :math:`W = Y`, and a constant :math:`W` (which reproduces only
    an independent pair).
    """
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    m, n = pxy.shape
    return (
        np.vstack([np.eye(m), (pxy / py).T, px[np.newaxis]]),
        np.vstack([pxy / px[:, np.newaxis], np.eye(n), py[np.newaxis]]),
    )


def _usable_pairs(pxy, px_w, py_w):
    """
    The distinct conditional pairs that never meet a zero of `pxy` (any other
    pair has infinite maximal cross entropy), and the products
    :math:`p(x|w) p(y|w)` as columns.
    """
    zero = (pxy <= 0).astype(float)
    usable = ((px_w > 0) @ zero * (py_w > 0)).sum(axis=1) == 0
    pairs = np.unique(np.round(np.hstack([px_w[usable], py_w[usable]]), 12), axis=0)
    px_w, py_w = pairs[:, : pxy.shape[0]], pairs[:, pxy.shape[0] :]
    products = (px_w[:, :, np.newaxis] * py_w[:, np.newaxis, :]).reshape(len(pairs), -1).T
    return px_w, py_w, products


def _reproduction(pxy, products):
    """The equality constraints asking a mixture of `products` to reproduce `pxy`."""
    k = products.shape[1]
    return np.vstack([products[:-1], np.ones((1, k))]), np.append(pxy.ravel()[:-1], 1.0)


def _exact_weights(pxy, products, weights):
    """
    Correct the mixture's weights so they meet its equality constraints to
    rounding rather than to the solver's feasibility tolerance. The correction is
    the smallest change to the used weights that does so, which keeps any balance
    the solver struck between them (an optimal mixture can use one more state
    than there are constraints, so re-solving from scratch would not).
    """
    used = weights > _MASS_TOL
    a_eq, b_eq = _reproduction(pxy, products[:, used])
    correction, *_ = np.linalg.lstsq(a_eq, b_eq - a_eq @ weights[used], rcond=None)
    exact = weights[used] + correction
    if (exact < -_MASS_TOL).any():  # pragma: no cover
        return weights
    out = np.zeros_like(weights)
    out[used] = np.clip(exact, 0.0, None)
    return out


def _witness(pxy, px_w, py_w, products, weights, tol=_MATCH_TOL):
    """
    The decomposition ``(p(w|x), p(y|w))`` of a mixture, indexed by codes, or
    None if it does not reproduce `pxy` to within `tol`.
    """
    used = weights > _MASS_TOL
    if np.abs(products[:, used] @ weights[used] - pxy.ravel()).max() > tol:  # pragma: no cover
        return None
    return (px_w[used] * weights[used, np.newaxis]).T / pxy.sum(axis=1)[:, np.newaxis], py_w[used]


def _mixture(pxy, px_w, py_w, objective, input_weight=0.0):
    r"""
    The best mixture of the conditional pairs ``(px_w[i], py_w[i])`` that
    reproduces `pxy`. With the pairs fixed, the sum rates are linear in the
    weights :math:`p(w)`, as is the constraint
    :math:`\sum_w p(w) p_{X|w} p_{Y|w} = p_{XY}`; the lower bound's coupling of
    :math:`p_W` with itself becomes extra variables. So the best mixture solves a
    linear program, whose basic solutions use at most :math:`|X||Y|` states.

    Parameters
    ----------
    pxy : np.ndarray
        The distribution to reproduce.
    px_w : np.ndarray, shape (k, |X|)
        Candidate conditionals :math:`p(x|w)`.
    py_w : np.ndarray, shape (k, |Y|)
        Candidate conditionals :math:`p(y|w)`.
    objective : str
        ``"upper"``, ``"lower"``, or ``"wyner"`` (:math:`I[X, Y : W]`).
    input_weight : float
        With ``"upper"``, minimize :math:`(1 - \lambda) \Gamma_W + \lambda I[X : W]`
        for this :math:`\lambda`.

    Returns
    -------
    result : (float, (np.ndarray, np.ndarray)) or None
        The objective and the decomposition ``(p(w|x), p(y|w))`` indexed by
        codes, or None if no mixture reproduces `pxy`.
    """
    px_w, py_w, products = _usable_pairs(pxy, px_w, py_w)
    k = len(px_w)
    if k == 0:
        return None
    a_eq, b_eq = _reproduction(pxy, products)
    input_entropies = _entropies(px_w)
    entropies = input_entropies + _entropies(py_w)
    cost = _cost_matrix(pxy)
    constant = 0.0

    if objective == "upper":
        c = (1 - input_weight) * (_max_coupling(px_w, py_w, cost)[0] - entropies) - input_weight * input_entropies
        constant = input_weight * float(_entropies(pxy.sum(axis=1)[np.newaxis])[0])
        bounds = (0, None)
    elif objective == "wyner":
        c = -entropies
        constant = float(_entropies(pxy.reshape(1, -1))[0])
        bounds = (0, None)
    else:
        zero = (pxy <= 0).astype(float)
        pair_costs = _max_coupling(np.repeat(px_w, k, axis=0), np.tile(py_w, (k, 1)), cost)[0]
        meets_zero = (((px_w > 0).astype(float) @ zero @ (py_w > 0).T.astype(float)) > 0).ravel()
        c = np.concatenate([-entropies, pair_costs])
        coupling_sums = np.vstack([np.kron(np.eye(k), np.ones((1, k))), np.kron(np.ones((1, k)), np.eye(k))[:-1]])
        a_eq = np.block(
            [
                [a_eq, np.zeros((len(a_eq), k * k))],
                [-np.vstack([np.eye(k), np.eye(k)[:-1]]), coupling_sums],
            ]
        )
        b_eq = np.concatenate([b_eq, np.zeros(2 * k - 1)])
        upper = np.concatenate([np.full(k, np.inf), np.where(meets_zero, 0.0, np.inf)])
        bounds = np.column_stack([np.zeros_like(upper), upper])

    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs", options=_LP_OPTIONS)
    if res.status != 0:  # pragma: no cover
        return None
    if objective == "lower":
        # A lower bound may err low, so a loosely matching mixture will do.
        witness = _witness(pxy, px_w, py_w, products, res.x[:k], tol=_LOOSE_TOL)
        return None if witness is None else (float(res.fun), witness)
    weights = _exact_weights(pxy, products, res.x[:k])
    witness = _witness(pxy, px_w, py_w, products, weights)
    if witness is None:  # pragma: no cover
        return None
    return float(c @ weights) + constant, witness


def _synthesis_mixture(pxy, px_w, py_w, common_randomness):
    r"""
    The smallest exact channel synthesis rate
    :math:`\max\{I[X : W], \Gamma_W - R_0\}` over mixtures of the conditional
    pairs that reproduce `pxy`, a linear program in the weights and the rate.

    Returns
    -------
    result : (float, (np.ndarray, np.ndarray), float) or None
        The rate, its decomposition ``(p(w|x), p(y|w))`` indexed by codes, and
        the weight :math:`\lambda` of :math:`I[X : W]` in the Lagrangian
        :math:`(1 - \lambda) \Gamma_W + \lambda I[X : W]` that the optimal
        mixture minimizes (the dual of the rate's first constraint); or None if
        no mixture reproduces `pxy`.
    """
    px_w, py_w, products = _usable_pairs(pxy, px_w, py_w)
    k = len(px_w)
    if k == 0:  # pragma: no cover
        return None
    a_eq, b_eq = _reproduction(pxy, products)
    input_entropies = _entropies(px_w)
    sum_rates = _max_coupling(px_w, py_w, _cost_matrix(pxy))[0] - input_entropies - _entropies(py_w)
    input_entropy = float(_entropies(pxy.sum(axis=1)[np.newaxis])[0])

    # Variables: the weights, then the rate t; minimize t >= I[X : W] and t >= Gamma_W - R_0.
    res = linprog(
        np.append(np.zeros(k), 1.0),
        A_ub=np.vstack([np.append(-input_entropies, -1.0), np.append(sum_rates, -1.0)]),
        b_ub=np.array([-input_entropy, common_randomness]),
        A_eq=np.hstack([a_eq, np.zeros((len(a_eq), 1))]),
        b_eq=b_eq,
        bounds=[(0, None)] * k + [(None, None)],
        method="highs",
        options=_LP_OPTIONS,
    )
    if res.status != 0:  # pragma: no cover
        return None
    weights = _exact_weights(pxy, products, res.x[:k])
    witness = _witness(pxy, px_w, py_w, products, weights)
    if witness is None:  # pragma: no cover
        return None
    rate = max(input_entropy - float(input_entropies @ weights), float(sum_rates @ weights) - common_randomness)
    weight = float(np.clip(-res.ineqlin.marginals[0], 0.0, 1.0))
    return rate, witness, weight


def _optimize(opt, pxy, seeds, niter, maxiter, polish, pool=None):
    """
    Minimize `opt`'s sum rate from `seeds` and random starts, refine each seed
    and the optimum to match the distribution, and return the best mixture of
    every state found (plus any in `pool`).

    Returns
    -------
    value : float
        The sum rate of the best mixture.
    decomposition : (np.ndarray, np.ndarray)
        Its ``(p(w|x), p(y|w))``, indexed by codes.
    pool : (np.ndarray, np.ndarray)
        The conditional pairs considered.
    """
    opt._seeds = [s for s in seeds if s is not None]
    candidates = list(opt._seeds)
    try:
        opt.optimize(niter=niter, maxiter=maxiter, polish=polish)
        candidates.append(opt._optima)
    except OptimizationException:  # pragma: no cover
        if not candidates:
            raise
    refined = [y for x in candidates for y in _refine(opt, x, maxiter)]
    vectors = candidates + refined
    # The trivial decompositions keep the mixture's linear program feasible.
    pairs = [_pairs(pxy, *_decomposition(opt, x)) for x in vectors] + [_trivial_pairs(pxy)]
    if pool is not None:
        pairs.append(pool)
    pool = (np.vstack([p for p, _ in pairs]), np.vstack([q for _, q in pairs]))

    result = _mixture(pxy, pool[0], pool[1], "lower" if opt._outer else "upper", input_weight=opt._input_weight)
    if result is None:  # pragma: no cover
        # A loosely matching decomposition could undercut the true minimum, so only refined ones may stand in.
        if not refined:
            msg = "No decomposition reproducing the distribution was found."
            raise ditException(msg)
        best = min(refined, key=lambda x: float(opt.objective(x)))
        result = float(opt.objective(best)), _decomposition(opt, best)
    return result[0], result[1], pool


def _check_pair(dist, rvs):
    """Resolve `rvs` to two sorted groups of variables."""
    if rvs is None:
        if dist.outcome_length() != 2:
            msg = "The exact common information rate needs `rvs` unless the distribution has exactly two variables."
            raise ditException(msg)
        rvs = [[0], [1]]
    if len(rvs) != 2:
        msg = f"The exact common information rate is defined for two groups of variables, not {len(rvs)}."
        raise ditException(msg)
    rvs, _ = normalize_rvs(dist, rvs, None)
    return [sorted(rv) for rv in rvs]


def _rate_bounds(dist, rvs, block_length, niter, maxiter, polish, bound, lower):
    """
    The bounds on the exact common information rate, in bits.

    Returns
    -------
    lower : float, None
        The lower bound, if requested.
    upper : float
        The smallest upper bound over block lengths up to `block_length`.
    """
    if int(block_length) != block_length or block_length < 1:
        msg = f"block_length must be a positive integer, not {block_length!r}."
        raise ditException(msg)
    pxy = _pair_matrix(dist, _check_pair(dist, rvs))
    pair = _pair_distribution(pxy)

    # Every common information lies between the mutual information and the joint entropy, and vanishes
    # exactly when the pair is independent.
    mutual_information = dual_total_correlation(pair, [[0], [1]])
    joint_entropy = float(-np.sum(pxy[pxy > 0] * np.log2(pxy[pxy > 0])))
    if np.isclose(mutual_information, 0.0) or np.isclose(mutual_information, joint_entropy):
        return (mutual_information if lower else None), mutual_information

    wyner = WynerCommonInformation(pair, [[0], [1]], bound=bound)
    wyner.optimize(niter=niter, maxiter=maxiter, polish=polish)
    exact = ExactCommonInformation(pair, [[0], [1]], bound=bound)
    exact.optimize(niter=niter, maxiter=maxiter, polish=polish)

    opt = ExactCommonInformationRate(pair, [[0], [1]], bound=bound)
    seeds = [_vector(opt, *_decomposition(other, other._optima)) for other in (wyner, exact)]
    value, witness, pool = _optimize(opt, pxy, seeds, niter, maxiter, polish)
    uppers, witnesses = [value], [witness]

    pxy_n = pxy
    for n in range(2, int(block_length) + 1):
        pxy_n = np.kron(pxy_n, pxy)
        opt_n = ExactCommonInformationRate(_pair_distribution(pxy_n), [[0], [1]], bound=bound)
        (p_w_a, p_y_a), (p_w_b, p_y_b) = witnesses[-1], witnesses[0]
        product = np.kron(p_w_a, p_w_b), np.kron(p_y_a, p_y_b)
        # The product's states join the pool, so the block can only improve on the shorter blocks.
        value_n, witness_n, _ = _optimize(
            opt_n, pxy_n, [_vector(opt_n, *product)], niter, maxiter, polish, pool=_pairs(pxy_n, *product)
        )
        uppers.append(value_n / n)
        witnesses.append(witness_n)
    upper = min(uppers)

    if not lower:
        return None, upper

    outer = ExactCommonInformationRate(pair, [[0], [1]], bound=bound, outer=True)
    seeds = [_vector(outer, *witness), _vector(outer, *_decomposition(wyner, wyner._optima))]
    outer_value, _, pool = _optimize(outer, pxy, seeds, niter, maxiter, polish, pool=pool)

    # The Wyner common information, minimized over the optimizer and every mixture of the states found.
    wyner_values = [float(wyner.objective(wyner._optima))]
    mixture = _mixture(pxy, pool[0], pool[1], "wyner")
    if mixture is not None:
        wyner_values.append(mixture[0])

    return min(max(min(wyner_values), outer_value), upper), upper


def _exact_channel_synthesis_rate(dist, rvs, common_randomness, niter, maxiter, polish, bound, rounds=5):
    r"""
    The Yu-Tan achievable rate for exact channel synthesis, in bits,

    .. math::

       R(R_0) = \min_{X - W - Y} \max\{I[X : W], \Gamma_W - R_0\},

    by column generation: the best mixture of the states found so far solves a
    linear program whose dual weights :math:`I[X : W]` by some :math:`\lambda`;
    minimizing :math:`(1 - \lambda) \Gamma_W + \lambda I[X : W]` over :math:`W`
    then supplies new states, until the rate stops improving.
    """
    pxy = _pair_matrix(dist, _check_pair(dist, rvs))
    pair = _pair_distribution(pxy)
    if common_randomness <= 0:
        # Gamma_W >= I[X, Y : W] >= I[X : W], so without common randomness the rate is the sum rate's minimum.
        return _rate_bounds(dist, rvs, 1, niter, maxiter, polish, bound, lower=False)[1]
    pool = _trivial_pairs(pxy)
    best = _synthesis_mixture(pxy, pool[0], pool[1], common_randomness)

    mutual_information = dual_total_correlation(pair, [[0], [1]])
    if np.isclose(best[0], mutual_information):
        return best[0]

    seeds = []
    for cls in (WynerCommonInformation, ExactCommonInformation):
        other = cls(pair, [[0], [1]], bound=bound)
        other.optimize(niter=niter, maxiter=maxiter, polish=polish)
        seeds.append(_decomposition(other, other._optima))

    for _ in range(rounds):
        rate, witness, weight = best
        opt = ExactCommonInformationRate(pair, [[0], [1]], bound=bound, input_weight=weight)
        vectors = [_vector(opt, *decomposition) for decomposition in [witness, *seeds]]
        _, _, pool = _optimize(opt, pxy, vectors, niter, maxiter, polish, pool=pool)
        seeds = []  # the common information optima are in the pool now
        candidate = _synthesis_mixture(pxy, pool[0], pool[1], common_randomness)
        if candidate is None or candidate[0] > rate - 1e-9:
            break
        best = candidate
    return max(best[0], mutual_information)


def _base(dist):
    """The numerical base of `dist`, or None if it is linear."""
    return dist.get_base(numerical=True) if dist.is_log() else None


def exact_common_information_rate(dist, rvs=None, block_length=1, niter=None, maxiter=1000, polish=1e-6, bound=None):
    r"""
    An upper bound on the exact common information rate :cite:`kumar2014exact`,
    from Yu and Tan :cite:`yu2020on,yu2020exact`:

    .. math::

       \bar{G}[X : Y] \leq \Gamma[X : Y]
         = \min_{X - W - Y} -\H{X, Y | W} + \sum_w p(w) \mathbb{H}(p_{X|w}, p_{Y|w} \| p_{XY}),

    where :math:`\mathbb{H}` is the :func:`~dit.divergences.maximal_cross_entropy`.
    Every feasible :math:`W` achieves its value, so the result is a valid upper
    bound even when the optimizer misses the minimum. It lies between the Wyner
    common information and the one-shot :func:`exact_common_information`,
    equals the Wyner common information when some optimal :math:`W` makes one of
    :math:`p_{X|w}, p_{Y|w}` deterministic for every :math:`w` (as for the
    erasure channel), and equals :math:`\bar{G}` for the doubly symmetric binary
    source.

    With ``block_length=N`` the bound is also applied to :math:`n` i.i.d. copies
    of the pair, returning :math:`\min_{n \leq N} \Gamma[X^n : Y^n] / n`, which
    decreases to :math:`\bar{G}` as :math:`N` grows. The work grows
    exponentially in :math:`N`; beyond small alphabets, pass `bound`.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list, None
        Two lists of indexes, the variables of :math:`X` and of :math:`Y`. If
        None, the distribution must have exactly two variables.
    block_length : int
        The largest number of i.i.d. copies to bound. Defaults to 1.
    niter : int > 0
        Number of basin hoppings to perform during each optimization.
    maxiter : int > 0
        The number of iterations of the optimization subroutine to perform.
    polish : False, float
        Whether to polish the result or not. If a float, this will perform a
        second optimization seeded with the result of the first, but with
        smaller tolerances and probabilities below polish set to 0. If False,
        don't polish.
    bound : int, None
        Bound the size of the auxiliary variable, for every block length.

    Returns
    -------
    rate : float
        The upper bound on the exact common information rate.

    Raises
    ------
    ditException
        Raised if `rvs` does not specify exactly two groups of variables.
    """
    _, upper = _rate_bounds(dist, rvs, block_length, niter, maxiter, polish, bound, lower=False)
    return _rescale(upper, _base(dist))


def exact_common_information_rate_bounds(
    dist, rvs=None, block_length=1, niter=None, maxiter=1000, polish=1e-6, bound=None
):
    r"""
    Lower and upper bounds on the exact common information rate
    :math:`\bar{G}[X : Y]` :cite:`kumar2014exact`.

    The upper bound is :func:`exact_common_information_rate`. The lower bound is
    the larger of the Wyner common information and the Yu-Tan bound
    :cite:`yu2020on,yu2020exact`

    .. math::

       \Gamma^{(o)}[X : Y] = \min_{X - W - Y} -\H{X, Y | W}
         + \min_{q \in \mathcal{C}(p_W, p_W)} \sum_{w, w'} q(w, w') \mathbb{H}(p_{X|w}, p_{Y|w'} \| p_{XY}).

    The two bounds meet for the doubly symmetric binary source. Unlike the upper
    bound, the lower bound holds only at a global minimum over auxiliary
    variables with at most `bound` states, which local optimization cannot
    certify, so treat it as an estimate.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list, None
        Two lists of indexes, the variables of :math:`X` and of :math:`Y`. If
        None, the distribution must have exactly two variables.
    block_length : int
        The largest number of i.i.d. copies used for the upper bound. Defaults
        to 1.
    niter : int > 0
        Number of basin hoppings to perform during each optimization.
    maxiter : int > 0
        The number of iterations of the optimization subroutine to perform.
    polish : False, float
        Whether to polish the result or not. If a float, this will perform a
        second optimization seeded with the result of the first, but with
        smaller tolerances and probabilities below polish set to 0. If False,
        don't polish.
    bound : int, None
        Bound the size of the auxiliary variable.

    Returns
    -------
    lower : float
        The lower bound.
    upper : float
        The upper bound.

    Raises
    ------
    ditException
        Raised if `rvs` does not specify exactly two groups of variables.
    """
    lower, upper = _rate_bounds(dist, rvs, block_length, niter, maxiter, polish, bound, lower=True)
    base = _base(dist)
    return _rescale(lower, base), _rescale(upper, base)
