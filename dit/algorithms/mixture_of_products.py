"""
Mixtures of fully factorized distributions (latent-class / naive-Bayes).

The family

.. math::

    \\mathcal{F}_k = \\Bigl\\{
        Q : Q(x) = \\sum_{\\alpha=1}^{k} \\pi_\\alpha
        \\prod_{i=1}^{n} Q_i(x_i \\mid \\alpha)
    \\Bigr\\}

is the standard representation underlying Wyner common information: variables
are independent given a discrete latent of cardinality :math:`k`.  Maximum
likelihood under :math:`P` is equivalent to the forward-KL projection

.. math::

    Q^{(k)} = \\arg\\min_{Q \\in \\mathcal{F}_k} D(P \\Vert Q)

and is fit by EM.  No support jitter is required: :math:`Q` may place mass
outside the support of a sparse :math:`P`.

See Rosas et al. (2019) for the shared-randomness / binding interpretation of
dual total correlation, and Wyner (1975) / Abdallah & Plumbley (2012) for the
common-information side.
"""

from copy import deepcopy
from itertools import product

import numpy as np

from ..distribution import Distribution
from .optutil import prepare_dist

__all__ = (
    "fit_mixture_of_products",
    "mixture_of_products_dists",
)


def _dense_table(dist):
    """
    Expand ``dist`` onto its Cartesian sample space.

    Returns
    -------
    outcomes : list of tuple
    pmf : ndarray, shape (n_outcomes,)
    X : ndarray, shape (n_outcomes, n_vars), integer-coded symbols
    sizes : list of int
    """
    d = prepare_dist(deepcopy(dist))
    n = d.outcome_length()
    alphabets = [tuple(sorted({o[i] for o in d.outcomes})) for i in range(n)]
    outcomes = list(product(*alphabets))
    pmf_map = {tuple(o): float(p) for o, p in zip(d.outcomes, d.pmf, strict=True)}
    pmf = np.array([pmf_map.get(o, 0.0) for o in outcomes], dtype=float)
    total = pmf.sum()
    if total <= 0:
        msg = "Distribution has no mass."
        raise ValueError(msg)
    pmf /= total
    sym_index = [{s: j for j, s in enumerate(a)} for a in alphabets]
    X = np.array([[sym_index[i][o[i]] for i in range(n)] for o in outcomes], dtype=int)
    sizes = [len(a) for a in alphabets]
    return outcomes, pmf, X, sizes


def _component_logprob(pi, conds, X):
    """Log joint component densities ``log(π_α ∏_i Q_i(x_i|α))``, shape (k, n_out)."""
    log_comp = np.log(pi + 1e-300)[:, None]
    for i, cond in enumerate(conds):
        log_comp = log_comp + np.log(cond[:, X[:, i]] + 1e-300)
    return log_comp


def _em_once(pmf, X, sizes, k, *, max_iter, tol, rng):
    """Single EM run. Returns (loglik, q_pmf, pi, conds, I_xv, H_v)."""
    n_out, n = X.shape
    pi = rng.dirichlet(np.ones(k))
    conds = [rng.dirichlet(np.ones(s), size=k) for s in sizes]
    prev_ll = -np.inf

    for _ in range(max_iter):
        log_comp = _component_logprob(pi, conds, X)
        m = log_comp.max(axis=0, keepdims=True)
        comp = np.exp(log_comp - m)
        r = comp / (comp.sum(axis=0, keepdims=True) + 1e-300)

        w = r * pmf[None, :]
        pi = w.sum(axis=1)
        pi = pi / (pi.sum() + 1e-300)

        for i in range(n):
            s = sizes[i]
            c = np.zeros((k, s))
            for a in range(k):
                for v in range(s):
                    c[a, v] = w[a, X[:, i] == v].sum()
                c[a] /= c[a].sum() + 1e-300
            conds[i] = c

        log_comp = _component_logprob(pi, conds, X)
        m = log_comp.max(axis=0)
        ll = float(np.sum(pmf * (m + np.log(np.exp(log_comp - m).sum(axis=0) + 1e-300))))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    log_comp = _component_logprob(pi, conds, X)
    m = log_comp.max(axis=0)
    q = np.exp(m) * np.exp(log_comp - m).sum(axis=0)
    q = q / q.sum()

    # Responsibilities under the data for I(X; V).
    m = log_comp.max(axis=0, keepdims=True)
    r = np.exp(log_comp - m)
    r = r / (r.sum(axis=0, keepdims=True) + 1e-300)
    p_a = (r * pmf[None, :]).sum(axis=1)
    p_a = p_a / (p_a.sum() + 1e-300)
    hv = float(-np.sum(p_a[p_a > 0] * np.log2(p_a[p_a > 0])))
    hv_x = 0.0
    for t in range(n_out):
        if pmf[t] <= 0:
            continue
        rt = r[:, t]
        rt = rt[rt > 0]
        hv_x += float(pmf[t] * (-np.sum(rt * np.log2(rt))))
    ixv = hv - hv_x

    return ll, q, pi, conds, float(ixv), hv


def fit_mixture_of_products(
    dist,
    k,
    *,
    n_init=12,
    max_iter=200,
    tol=1e-10,
    seed=0,
):
    """
    Fit a :math:`k`-mixture of product distributions to ``dist`` by EM.

    Parameters
    ----------
    dist : Distribution
        Target joint.
    k : int
        Number of mixture components.
    n_init, max_iter : int
        Random restarts and EM iteration cap.
    tol : float
        Log-likelihood convergence tolerance.
    seed : int
        RNG seed for restarts.

    Returns
    -------
    result : dict
        Keys ``dist`` (fitted :class:`Distribution`), ``pi``, ``conds``,
        ``I_xv`` (:math:`I(X;V)` under data-weighted responsibilities),
        ``H_v``, ``loglik``.
    """
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)

    outcomes, pmf, X, sizes = _dense_table(dist)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        run_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        cand = _em_once(pmf, X, sizes, k, max_iter=max_iter, tol=tol, rng=run_rng)
        if best is None or cand[0] > best[0]:
            best = cand

    ll, q_pmf, pi, conds, ixv, hv = best
    q = Distribution(outcomes, q_pmf, base="linear", validate=False)
    q.normalize()
    return {
        "dist": q,
        "pi": pi,
        "conds": conds,
        "I_xv": ixv,
        "H_v": hv,
        "loglik": ll,
    }


def mixture_of_products_dists(
    dist,
    k_max=None,
    *,
    n_init=12,
    max_iter=200,
    tol=1e-10,
    seed=0,
    early_stop=True,
    kl_tol=1e-8,
):
    """
    Fit the mixture-of-products ladder :math:`Q^{(1)},\\ldots,Q^{(k_{\\max})}`.

    Parameters
    ----------
    dist : Distribution
    k_max : int or None
        Maximum number of components.  Default ``min(8, |X|)``.
    n_init, max_iter, tol, seed
        Passed to :func:`fit_mixture_of_products` (seed offset by ``k``).
    early_stop : bool
        If True, stop once :math:`D(P\\Vert Q^{(k)}) <` ``kl_tol``.
    kl_tol : float
        Forward-KL threshold for early stopping.

    Returns
    -------
    dists : list of Distribution
        ``dists[k-1]`` is the MLE in :math:`\\mathcal{F}_k`.
    meta : list of dict
        Per-``k`` diagnostics (``I_xv``, ``H_v``, ``loglik``, ``pi``, ``conds``).
    """
    from ..divergences import kullback_leibler_divergence as D

    outcomes, pmf, _, _ = _dense_table(dist)
    p_dense = Distribution(outcomes, pmf, base="linear", validate=False)
    p_dense.normalize()

    if k_max is None:
        k_max = min(8, len(outcomes))
    k_max = max(1, int(k_max))

    dists = []
    meta = []
    for k in range(1, k_max + 1):
        fit = fit_mixture_of_products(
            dist,
            k,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            seed=seed + 17 * k,
        )
        dists.append(fit["dist"])
        entry = {key: fit[key] for key in ("I_xv", "H_v", "loglik", "pi", "conds")}
        entry["forward_kl"] = float(D(p_dense, fit["dist"]))
        meta.append(entry)
        if early_stop and entry["forward_kl"] < kl_tol:
            break

    return dists, meta
