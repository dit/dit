"""
The W-information and M-information of Liardi et al.

For sources :math:`X = (X_0, \\ldots, X_{n-1})` and targets :math:`Y = (Y_0, \\ldots, Y_{m-1})`,
the W-information is the least mutual information among the distributions that share
every source-target pairwise marginal with :math:`p` :cite:`liardi2025scalable`,

.. math::

   \\mathcal{W}(X : Y) = \\min_{q \\in \\Delta_p} \\I{X : Y}_q, \\qquad
   \\Delta_p = \\{q : q(x_i, y_j) = p(x_i, y_j) \\text{ for all } i, j\\},

and the M-information is the rest, :math:`\\mathcal{M}(X : Y) = \\I{X : Y} - \\mathcal{W}(X : Y)`:
the part of the mutual information that no distribution with the same source-target pairs
can do without. Only the source-target pairs are kept, not the pairs within the sources or
within the targets. With a single target, :math:`\\mathcal{W}` is the union information of
Bertschinger et al. and Griffith & Koch :cite:`bertschinger2014quantifying,griffith2014quantifying`;
for two sources it is :math:`\\I{X_0 : Y} + \\I{X_1 : Y} - \\Ibroja{X_0, X_1 : Y}`, so that
:math:`\\mathcal{M}` is the BROJA synergy.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp, softmax

from ..helpers import parse_rvs
from ..utils import unitful

__all__ = (
    "m_information",
    "w_information",
)

_LN2 = np.log(2)
# values of X or Y whose mass falls below this while still shrinking are dropped
_DROP = 1e-2
# L-BFGS keeps the logarithms of the masses on each side within this of each other
_SPREAD = np.log(100)
# the most Newton steps per projection
_MAX_NEWTON = 100
# the alternating projections that each start gets before the best one is polished further
_BRIEF = 200


def _groups(dist, sources, targets):
    """
    Resolve the source and target groups to tuples of variable indices.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources, targets : iter of iters, None
        The source and target groups. If both are None, the first half of the variables
        are the sources and the second half the targets.

    Returns
    -------
    sources, targets : list of tuples
        The groups, as tuples of variable indices.

    Raises
    ------
    ValueError
        If only one of `sources` and `targets` is given, a group is empty, or two groups
        share a variable.
    """
    if sources is None and targets is None:
        n = dist.outcome_length()
        if n % 2:
            raise ValueError("Specify `sources` and `targets` for an odd number of variables.")
        sources, targets = [[i] for i in range(n // 2)], [[i] for i in range(n // 2, n)]
    elif sources is None or targets is None:
        raise ValueError("Specify both `sources` and `targets`, or neither.")
    sources = [tuple(parse_rvs(dist, list(group), sort=False)[1]) for group in sources]
    targets = [tuple(parse_rvs(dist, list(group), sort=False)[1]) for group in targets]
    groups = sources + targets
    if not sources or not targets or not all(groups):
        raise ValueError("The sources and targets must be nonempty groups of variables.")
    variables = [v for group in groups for v in group]
    if len(set(variables)) != len(variables):
        raise ValueError("The source and target groups must not share variables.")
    return sources, targets


def _joint(dist, groups):
    """
    The pmf of `dist` as a dense array with one axis per group.
    """
    d = dist.coalesce([list(group) for group in groups])
    d.make_dense()
    shape = [len(alphabet) for alphabet in d.alphabet]
    return np.asarray(d.pmf, dtype=float).reshape(shape)


def _mutual_information(q, n):
    """
    I(X : Y) in bits, where X is the first `n` axes of `q` and Y the others.
    """
    qx = q.sum(axis=tuple(range(n, q.ndim)), keepdims=True)
    qy = q.sum(axis=tuple(range(n)), keepdims=True)
    mask = q > 0
    # a difference of logarithms, since q(x) q(y) underflows when both are tiny
    with np.errstate(divide="ignore"):
        terms = np.log(q) - np.log(qx) - np.log(qy)
    return float(np.sum(q[mask] * terms[mask]) / _LN2)


def _gap(q, log_m, values):
    """
    max_v log m(v) - sum_v q(v) log m(v), in nats, the maximum taken over `values`.

    Here q = r m is a marginal of an iterate and m = q / r are its growth factors, which stay
    defined where r(v) = 0. Taken over every value, it is zero exactly at a fixed point whose
    missing values could not grow; when the other side's marginal is fixed, it bounds the
    distance to the minimum, by duality.
    """
    support = q > 0
    return float(log_m[values].max() - np.dot(q[support], log_m[support]))


def _support(p, allowed, constraints):
    """
    The outcomes to which some distribution with the pairwise marginals of `p` gives positive
    probability.

    Every outcome of `p` is one. An outcome that the pairwise marginals allow can still have
    zero probability under all of those distributions; the projection of a positive start is
    then only approached, with diverging multipliers. So each other allowed outcome is checked
    by a linear program, which maximizes its probability.

    Parameters
    ----------
    p : np.ndarray
        The joint pmf.
    allowed : np.ndarray
        The outcomes whose pairwise marginals are all positive.
    constraints : list of (tuple, np.ndarray)
        The axes summed out for each pairwise marginal, and the marginal (with `keepdims`).

    Returns
    -------
    support : np.ndarray
        The outcomes some distribution in the polytope gives positive probability.
    """
    support = allowed & (p > 0)
    if (support == allowed).all():
        return support
    cells = np.flatnonzero(allowed)
    coords = np.unravel_index(cells, p.shape)
    rows, rhs, offset = [], [], 0
    for axes, marginal in constraints:
        i, j = (k for k in range(p.ndim) if k not in axes)
        values, row = np.unique(coords[i] * p.shape[j] + coords[j], return_inverse=True)
        rows.append(row + offset)
        rhs.append(marginal.ravel()[values])
        offset += len(values)
    rows = np.concatenate(rows)
    columns = np.tile(np.arange(len(cells)), len(constraints))
    matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(offset, len(cells)))
    options = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
    possible = support.ravel()[cells]
    for k in np.flatnonzero(~possible):
        if not possible[k]:
            objective = np.zeros(len(cells))
            objective[k] = -1
            result = linprog(objective, A_eq=matrix, b_eq=np.concatenate(rhs), method="highs", options=options)
            # every outcome that a distribution in the polytope gives mass to is possible
            possible |= result.x > 1e-8
    support = np.zeros(p.size, dtype=bool)
    support[cells[possible]] = True
    return support.reshape(p.shape)


class _Polytope:
    """
    The distributions that share the source-target pairwise marginals of a joint pmf, and the
    minimization of the mutual information between the sources and the targets over them.

    Distributions are vectors over the outcomes some distribution of the polytope gives positive
    probability, and the marginals r_x, r_y vectors over the values of X and of Y.
    """

    def __init__(self, p, n):
        """
        Parameters
        ----------
        p : np.ndarray
            The joint pmf, with the `n` source groups as the first axes and the targets after.
        n : int
            The number of source groups.
        """
        ndim, self.n = p.ndim, n
        pairs = [(i, j) for i in range(n) for j in range(n, ndim)]
        constraints = []
        for i, j in pairs:
            axes = tuple(k for k in range(ndim) if k not in (i, j))
            constraints.append((axes, p.sum(axis=axes, keepdims=True)))
        allowed = np.ones(p.shape, dtype=bool)
        for _, marginal in constraints:
            allowed &= marginal > 0
        self.allowed = _support(p, allowed, constraints)
        coords = np.nonzero(self.allowed)
        # each outcome has one feature per pair: the index of its value of that pair
        columns, targets, offset = [], [], 0
        for (i, j), (_, marginal) in zip(pairs, constraints, strict=True):
            marginal = marginal.reshape(p.shape[i], p.shape[j])
            index = np.full(marginal.shape, -1)
            support = marginal > 0
            index[support] = np.arange(offset, offset + support.sum())
            columns.append(index[coords[i], coords[j]])
            targets.append(marginal[support])
            offset += support.sum()
        self.features = np.zeros((len(coords[0]), offset))
        self.features[np.arange(len(coords[0]))[:, None], np.stack(columns, axis=1)] = 1
        self.targets = np.concatenate(targets)
        self.size_x, self.size_y = int(np.prod(p.shape[:n])), int(np.prod(p.shape[n:]))
        self.x = np.ravel_multi_index(coords[:n], p.shape[:n])
        self.y = np.ravel_multi_index(coords[n:], p.shape[n:])
        # the values of X (of Y) with at least one possible outcome
        self.values_x = np.bincount(self.x, minlength=self.size_x) > 0
        self.values_y = np.bincount(self.y, minlength=self.size_y) > 0
        self.upper = _mutual_information(p, n)

    def marginals(self, q):
        return np.bincount(self.x, q, self.size_x), np.bincount(self.y, q, self.size_y)

    def mutual_information(self, q):
        """
        I(X : Y) of `q`, in bits.
        """
        q_x, q_y = self.marginals(q)
        support = q > 0
        q, x, y = q[support], self.x[support], self.y[support]
        # a difference of logarithms, since q(x) q(y) underflows when both are tiny
        return float(np.dot(q, np.log(q) - np.log(q_x[x]) - np.log(q_y[y])) / _LN2)

    def project(self, r_x, r_y, lam):
        """
        The I-projection of r_x r_y onto the polytope, by Newton's method on its dual: the
        projection is q = r_x r_y exp(features @ lam) / Z for the multipliers `lam` of the pair
        values that minimize log Z - targets @ lam. It starts from the given multipliers, and
        once more from zero if that fails.

        Returns
        -------
        q : np.ndarray, None
            The projection, or None if Newton's method did not converge, which happens when the
            support of r_x r_y cannot carry the marginals, or carries them only in the limit.
        lam : np.ndarray
            The multipliers.
        kernel : np.ndarray
            exp(features @ lam) / Z, so that q = r_x r_y kernel; it is defined on every possible
            outcome, including those where r_x r_y = 0.
        """
        with np.errstate(divide="ignore"):
            log_r = np.log(r_x)[self.x] + np.log(r_y)[self.y]
        live = np.isfinite(log_r)
        if not self.features[live].any(axis=0).all():
            # some pair value has no outcome left
            return None, lam, None
        q, new_lam, kernel = self._newton(log_r, live, lam)
        if q is None and lam.any():
            q, new_lam, kernel = self._newton(log_r, live, np.zeros_like(lam))
        return q, new_lam, kernel

    def _newton(self, log_r, live, lam):
        """
        Newton's method for the multipliers of the projection of exp(log_r), from `lam`.
        """
        features, log_r, targets = self.features[live], log_r[live], self.targets
        for _ in range(_MAX_NEWTON):
            z = log_r + features @ lam
            log_z = logsumexp(z)
            q = np.exp(z - log_z)
            moments = features.T @ q
            grad = moments - targets
            if np.abs(grad).max() <= 1e-14:
                break
            hessian = (features.T * q) @ features - np.outer(moments, moments)
            step = np.linalg.lstsq(hessian, -grad, rcond=None)[0]
            decrease = -grad @ step
            if decrease < 1e-12:
                # a full step, since rounding would hide the decrease from a line search
                lam = lam + step
                continue
            value, size = log_z - targets @ lam, 1.0
            while True:
                trial = lam + size * step
                if (
                    logsumexp(log_r + features @ trial) - targets @ trial <= value - 1e-4 * size * decrease
                    or size < 1e-10
                ):
                    break
                size /= 2
            lam = trial
        z = log_r + features @ lam
        log_z = logsumexp(z)
        q = np.exp(z - log_z)
        if not np.isfinite(q).all() or np.abs(features.T @ q - targets).max() > 1e-12:
            return None, lam, None
        projection = np.zeros(len(live))
        projection[live] = q
        return projection, lam, np.exp(self.features @ lam - log_z)

    def descend(self, r_x, r_y, lam, maxiter):
        """
        Minimize F(r_x, r_y) = min_q D(q || r_x r_y), the least divergence of the polytope from
        r_x r_y, whose minimum over the marginals is W. It uses L-BFGS in the logarithms of the
        marginals, where the gradient of F is (r_x - q_x, r_y - q_y). The logarithms on each
        side stay within `_SPREAD` of each other; `polish` takes over from there.
        """
        ix, iy = np.flatnonzero(self.values_x), np.flatnonzero(self.values_y)
        state = {"lam": lam}

        def marginals(theta):
            r_x, r_y = np.zeros(self.size_x), np.zeros(self.size_y)
            r_x[ix], r_y[iy] = softmax(theta[: len(ix)]), softmax(theta[len(ix) :])
            return r_x, r_y

        def objective(theta):
            r_x, r_y = marginals(theta)
            q, lam, _ = self.project(r_x, r_y, state["lam"])
            if q is None:  # pragma: no cover
                return np.inf, np.zeros_like(theta)
            state["lam"] = lam
            q_x, q_y = self.marginals(q)
            support = q > 0
            value = np.dot(q[support], np.log(q[support]))
            value -= np.dot(q_x[ix], np.log(r_x[ix])) + np.dot(q_y[iy], np.log(r_y[iy]))
            return value, np.concatenate([r_x[ix] - q_x[ix], r_y[iy] - q_y[iy]])

        theta = []
        for r, index in ((r_x, ix), (r_y, iy)):
            log_r = np.log(np.maximum(r[index], 1e-300))
            theta.append(np.clip(log_r - log_r.max() + _SPREAD / 2, -_SPREAD / 2, _SPREAD / 2))
        bounds = [(-_SPREAD / 2, _SPREAD / 2)] * (len(ix) + len(iy))
        options = {"maxiter": maxiter, "ftol": 0, "gtol": 1e-12, "maxcor": 20}
        result = minimize(objective, np.concatenate(theta), jac=True, method="L-BFGS-B", bounds=bounds, options=options)
        return (*marginals(result.x), state["lam"])

    def polish(self, r_x, r_y, lam, tol, maxiter, memory=None):
        """
        Alternate the I-projection of r_x r_y onto the polytope with replacing r_x, r_y by the
        marginals of the projection, until the gap is below `tol`; each step lowers I(X : Y).

        A minimum that gives some values of X or Y no mass is only approached sublinearly. So a
        value whose mass falls below `_DROP` while shrinking is dropped. A dropped value is
        restored, and kept from then on, if it would grow at all once the rest has settled, or
        at once if it would more than e-fold.

        Returns
        -------
        value : float
            The least mutual information found, in bits; at most that of p, which is in the
            polytope.
        gap : float
            The gap at the last iterate, in bits.
        state : tuple
            The marginals, multipliers and `memory` reached, from which `polish` can continue.
        """
        values_x, values_y = self.values_x, self.values_y
        if memory is None:
            # which values are kept from being dropped, and the mass each value had when it was
            # last dropped, where it is restored
            keep_x, keep_y = np.zeros(self.size_x, dtype=bool), np.zeros(self.size_y, dtype=bool)
            last_x, last_y = np.full(self.size_x, _DROP), np.full(self.size_y, _DROP)
        else:
            keep_x, keep_y, last_x, last_y = memory
        best, gap = self.upper, np.inf
        for _ in range(maxiter):
            q, new_lam, kernel = self.project(r_x, r_y, lam)
            dropped_x, dropped_y = values_x & (r_x == 0), values_y & (r_y == 0)
            if q is None:
                if not (dropped_x.any() or dropped_y.any()):
                    break
                # a dropped value is needed after all
                keep_x |= dropped_x
                keep_y |= dropped_y
                r_x, r_y = np.where(dropped_x, last_x, r_x), np.where(dropped_y, last_y, r_y)
                continue
            lam = new_lam
            q_x, q_y = self.marginals(q)
            with np.errstate(divide="ignore"):
                log_m_x = np.log(np.bincount(self.x, r_y[self.y] * kernel, self.size_x))
                log_m_y = np.log(np.bincount(self.y, r_x[self.x] * kernel, self.size_y))
            best = min(best, self.mutual_information(q))
            gap = max(_gap(q_x, log_m_x, values_x), _gap(q_y, log_m_y, values_y)) / _LN2
            if gap <= tol:
                break
            active_x, active_y = values_x & (r_x > 0), values_y & (r_y > 0)
            settled = max(_gap(q_x, log_m_x, active_x), _gap(q_y, log_m_y, active_y)) / _LN2 <= tol
            # a dropped value is restored, and kept from then on, if it would grow at all once the
            # rest has settled, or at once if it would more than e-fold
            threshold = tol * _LN2 if settled else 1.0
            sides = (
                (q_x, log_m_x, dropped_x, keep_x, last_x),
                (q_y, log_m_y, dropped_y, keep_y, last_y),
            )
            for q_v, log_m, dropped, keep, last in sides:
                grow = dropped & (log_m > threshold)
                keep |= grow
                q_v[grow] = last[grow]
                drop = (q_v > 0) & (q_v < _DROP) & (log_m < 0) & ~keep
                last[drop] = q_v[drop]
                q_v[drop] = 0
            r_x, r_y = q_x, q_y
        return best, gap, (r_x, r_y, lam, (keep_x, keep_y, last_x, last_y))


def _w_information(p, n, niter=None, tol=1e-10, maxiter=10_000, rng=None):
    """
    The W-information of a joint pmf, in bits.

    Parameters
    ----------
    p : np.ndarray
        The joint pmf, with the `n` source groups as the first axes and the target groups
        as the others.
    n : int
        The number of source groups.
    niter : int, None
        The number of random starts when both sides have several groups. Defaults to 10.
    tol : float
        The gap, in bits, at which a run stops.
    maxiter : int
        The largest number of iterations of each phase of a run.
    rng : np.random.Generator, int, None
        The random generator (or seed) for the random starts.

    Returns
    -------
    w : float
        The W-information.
    """
    if n == 1 and p.ndim == 2:
        # the only distribution with the joint marginal is p itself
        return _mutual_information(p, n)
    polytope = _Polytope(p, n)
    lam = np.zeros(len(polytope.targets))
    r_x = polytope.values_x / polytope.values_x.sum()
    r_y = polytope.values_y / polytope.values_y.sum()
    if n == 1 or p.ndim - n == 1:
        # one side's marginal is fixed: convex, so the projections alone reach the minimum
        best, gap, _ = polytope.polish(r_x, r_y, lam, tol, maxiter)
    else:
        starts = [(r_x, r_y)]
        rng = np.random.default_rng(rng)
        for _ in range(10 if niter is None else niter):
            r_x = rng.dirichlet(np.ones(polytope.size_x)) * polytope.values_x
            r_y = rng.dirichlet(np.ones(polytope.size_y)) * polytope.values_y
            starts.append((r_x, r_y))
        # each start descends by L-BFGS, which crosses the flat valleys that the projections
        # creep along, and is polished briefly, enough to rank the minima found; the best is
        # then polished until the gap is below `tol`
        runs = []
        for r_x, r_y in starts:
            d_x, d_y, d_lam = polytope.descend(r_x, r_y, lam, maxiter)
            runs.append(polytope.polish(d_x, d_y, d_lam, tol, _BRIEF))
        best, gap, (r_x, r_y, lam, memory) = min(runs, key=lambda run: run[0])
        if gap > tol:
            value, gap, _ = polytope.polish(r_x, r_y, lam, tol, maxiter, memory)
            best = min(best, value)
    if gap > tol:
        msg = f"The W-information did not converge to {tol} bits within {maxiter} iterations (gap {gap:.3g} bits)."
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
    return best


@unitful
def w_information(dist, sources=None, targets=None, *, niter=None, tol=1e-10, maxiter=10_000, rng=None):
    """
    The W-information :math:`\\mathcal{W}(X : Y)` of Liardi et al. :cite:`liardi2025scalable`.

    It is the least mutual information between the sources and the targets among the
    distributions that share every source-target pairwise marginal :math:`p(x_i, y_j)` with
    `dist`; the pairs within the sources, or within the targets, are free. With a single
    target it is the union information of :cite:`bertschinger2014quantifying,griffith2014quantifying`.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources, targets : iter of iters, None
        The source and target groups of variables, e.g. ``[[0], [1]]`` and ``[[2], [3]]``.
        The groups must be disjoint. If both are None, the first half of the variables are
        the sources and the second half the targets, one group each.
    niter : int, None
        The number of random restarts when both the sources and the targets have more than
        one group. Defaults to 10.
    tol : float
        The gap, in bits, at which the minimization stops (see Notes).
    maxiter : int
        The largest number of iterations per start.
    rng : np.random.Generator, int, None
        The random generator (or seed) for the restarts.

    Returns
    -------
    w : float
        The W-information.

    Raises
    ------
    ValueError
        If only one of `sources` and `targets` is given, if a group is empty, or if two groups
        share a variable.

    Warns
    -----
    RuntimeWarning
        If no start reaches `tol` within `maxiter` iterations.

    Notes
    -----
    The minimum is found by alternating minimization :cite:`csiszar1984information`. Given
    marginals :math:`r_X, r_Y`, the distribution :math:`q` is the I-projection of
    :math:`r_X r_Y` onto the distributions with the source-target marginals of `dist`
    (computed by Newton's method on its dual), and :math:`r_X, r_Y` are then replaced by the
    marginals of :math:`q`. Every step lowers :math:`\\I{X : Y}_q`, and a run stops once
    :math:`\\max_x \\log \\frac{q(x)}{r(x)} - \\DKL{q \\| r}` is below `tol` on both sides.

    When the sources or the targets form a single group, its marginal is fixed, the problem
    is convex, and this quantity bounds the distance to the minimum: the result is within
    `tol` of :math:`\\mathcal{W}`. Otherwise neither marginal is fixed and the problem is not
    convex: the minimizers need not be unique, and there can be local minima that are not
    global, e.g. for three binary sources and three binary targets. The minimization then
    starts from the uniform marginals and from `niter` random ones, each first descending by
    L-BFGS in the logarithms of the marginals; the best of these runs is polished, and its
    value, which is an upper bound on :math:`\\mathcal{W}`, is returned.

    Examples
    --------
    Copying one bit forward leaves nothing beyond the pairs, while the xor of two bits is
    invisible to every source-target pair:

    >>> d = dit.Distribution(['0000', '1111'], [1/2, 1/2])
    >>> w_information(d)
    1.0
    >>> d = dit.Distribution(['000', '011', '101', '110'], [1/4] * 4)
    >>> w_information(d, [[0], [1]], [[2]])
    0.0
    """
    sources, targets = _groups(dist, sources, targets)
    p = _joint(dist, sources + targets)
    return _w_information(p, len(sources), niter=niter, tol=tol, maxiter=maxiter, rng=rng)


@unitful
def m_information(dist, sources=None, targets=None, *, niter=None, tol=1e-10, maxiter=10_000, rng=None):
    """
    The M-information :math:`\\mathcal{M}(X : Y) = \\I{X : Y} - \\mathcal{W}(X : Y)` of Liardi et
    al. :cite:`liardi2025scalable`.

    It is the part of the mutual information between the sources and the targets that every
    distribution with the same source-target pairwise marginals must also carry beyond them:
    the information in their higher-order dependencies. With a single target it is the
    synergy of the union information; for two sources, the BROJA synergy.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources, targets : iter of iters, None
        The source and target groups of variables. See :func:`w_information`.
    niter : int, None
        The number of random restarts. See :func:`w_information`.
    tol : float
        The gap, in bits, at which the minimization stops.
    maxiter : int
        The largest number of iterations per start.
    rng : np.random.Generator, int, None
        The random generator (or seed) for the restarts.

    Returns
    -------
    m : float
        The M-information. When both the sources and the targets have more than one group,
        it is a lower bound (see the Notes of :func:`w_information`).

    Examples
    --------
    >>> d = dit.Distribution(['000', '011', '101', '110'], [1/4] * 4)
    >>> m_information(d, [[0], [1]], [[2]])
    1.0
    """
    sources, targets = _groups(dist, sources, targets)
    p = _joint(dist, sources + targets)
    w = _w_information(p, len(sources), niter=niter, tol=tol, maxiter=maxiter, rng=rng)
    return _mutual_information(p, len(sources)) - w
