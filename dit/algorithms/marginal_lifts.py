"""
Convex combinations of lifts of a joint's own marginals.

At order :math:`k`, form building blocks

.. math::

    \\bigl\\{ U \\bigr\\} \\cup
    \\bigl\\{ \\mathrm{Lift}(P_S) : 1 \\le |S| \\le k \\bigr\\}

and fit nonnegative weights summing to one by least squares:

.. math::

    Q^{(k)} = \\arg\\min_{\\alpha \\ge 0,\\ \\sum \\alpha = 1}
    \\bigl\\| P - \\textstyle\\sum_S \\alpha_S \\mathrm{Lift}(P_S) \\bigr\\|_2^2.

Lifts:

* ``uniform`` — :math:`P_S \\otimes U_{X \\setminus S}`
* ``product`` — :math:`P_S \\otimes \\prod_{i \\notin S} P_i`
"""

from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize

from ..algorithms.optutil import prepare_dist
from ..distribution import Distribution
from ..exceptions import ditException

__all__ = (
    "lift_marginal",
    "fit_marginal_lift_mixture",
    "marginal_lift_dists",
)


def _cartesian(dist):
    d = prepare_dist(deepcopy(dist))
    n = d.outcome_length()
    alph = [tuple(sorted({o[i] for o in d.outcomes})) for i in range(n)]
    outs = list(product(*alph))
    pmf_map = {tuple(o): float(p) for o, p in zip(d.outcomes, d.pmf, strict=True)}
    pmf = np.array([pmf_map.get(o, 0.0) for o in outs], dtype=float)
    pmf /= pmf.sum()
    return outs, pmf, alph, d


def lift_marginal(outs, pmf, alph, S, mode="uniform"):
    """
    Lift a marginal on coordinates ``S`` to a full joint pmf.

    Parameters
    ----------
    outs, pmf, alph
        Dense Cartesian table for the joint.
    S : tuple of int
        Variable indices of the marginal.
    mode : {'uniform', 'product'}
        How to extend off ``S``.
    """
    n = len(alph)
    S = tuple(S)
    marg = defaultdict(float)
    for o, p in zip(outs, pmf, strict=True):
        marg[tuple(o[i] for i in S)] += p
    rest = [i for i in range(n) if i not in S]
    if mode == "uniform":
        rest_size = int(np.prod([len(alph[i]) for i in rest])) if rest else 1
        return np.array([marg[tuple(o[i] for i in S)] / rest_size for o in outs], dtype=float)
    if mode != "product":
        msg = f"unknown lift mode {mode!r}"
        raise ditException(msg)
    ones = []
    for i in range(n):
        m1 = defaultdict(float)
        for o, p in zip(outs, pmf, strict=True):
            m1[o[i]] += p
        ones.append(m1)
    q = np.empty(len(outs), dtype=float)
    for t, o in enumerate(outs):
        val = marg[tuple(o[i] for i in S)]
        for i in rest:
            val *= ones[i][o[i]]
        q[t] = val
    return q / q.sum()


def fit_marginal_lift_mixture(dist, order, mode="uniform", n_init=12, seed=0):
    """
    Fit a convex combination of lifts of marginals of order at most ``order``.

    Returns
    -------
    result : dict
        Keys ``dist``, ``labels``, ``alpha``, ``L2``.
    """
    if order < 0:
        msg = "order must be nonnegative"
        raise ditException(msg)
    outs, pmf, alph, template = _cartesian(dist)
    n = len(alph)
    order = min(order, n)

    blocks, labels = [], []
    blocks.append(np.ones(len(outs)) / len(outs))
    labels.append(())
    for k in range(1, order + 1):
        for S in combinations(range(n), k):
            blocks.append(lift_marginal(outs, pmf, alph, S, mode=mode))
            labels.append(S)
    A = np.column_stack(blocks)
    nb = A.shape[1]

    def loss(x):
        return float(np.sum((A @ x - pmf) ** 2))

    cons = {"type": "eq", "fun": lambda x: float(x.sum() - 1.0)}
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        x0 = rng.dirichlet(np.ones(nb))
        res = minimize(
            loss,
            x0,
            bounds=[(0.0, None)] * nb,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-14, "disp": False},
        )
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)

    alpha = np.maximum(best[1], 0.0)
    alpha = alpha / alpha.sum()
    q = A @ alpha
    q = np.maximum(q, 0.0)
    q = q / q.sum()
    qd = Distribution(outs, q, base="linear", validate=False)
    qd.normalize()
    if template.get_rv_names() is not None:
        qd.set_rv_names(template.get_rv_names())
    return {
        "dist": qd,
        "labels": labels,
        "alpha": alpha,
        "L2": float(np.sqrt(best[0])),
    }


def marginal_lift_dists(dist, k_max=None, mode="uniform", n_init=12, seed=0):
    """
    Ladder of marginal-lift mixtures for orders :math:`0,\\ldots,k_{\\max}`.

    Order 0 is the uniform distribution. Order :math:`n` includes the full
    joint as a block and recovers :math:`P` exactly.
    """
    n = dist.outcome_length()
    if k_max is None:
        k_max = n
    k_max = min(int(k_max), n)

    dists = []
    metas = []
    for k in range(0, k_max + 1):
        if k == 0:
            outs, _, _, template = _cartesian(dist)
            q = Distribution(outs, np.ones(len(outs)) / len(outs), base="linear", validate=False)
            q.normalize()
            if template.get_rv_names() is not None:
                q.set_rv_names(template.get_rv_names())
            dists.append(q)
            metas.append({"labels": [()], "alpha": np.array([1.0]), "L2": None})
        else:
            fit = fit_marginal_lift_mixture(dist, k, mode=mode, n_init=n_init, seed=seed + k)
            dists.append(fit["dist"])
            metas.append({key: fit[key] for key in ("labels", "alpha", "L2")})
    return dists, metas
