#!/usr/bin/env python3
"""
Probe: linear building-block approximations of a joint.

(1) Amari m-flat ANOVA family M_k, projected under forward KL / reverse KL / JSD.
(2) Convex combinations of lifts of the data's own marginals P_S.

Suite: Giant Bit, XOR, W, Copy, AND — plus BindingMixtureProfile as contrast.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize

from dit import Distribution
from dit.algorithms.mprojection import (
    _aligned_pmf,
    _alphabets,
    _dist_from_pmf,
    _outcome_tuples,
    _project_onto_mflat,
    mflat_design_matrix,
    symmetric_smooth,
)
from dit.algorithms.optutil import prepare_dist
from dit.divergences import kullback_leibler_divergence as D
from dit.divergences.jensen_shannon_divergence import jensen_shannon_divergence_pmf
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import total_correlation as T
from dit.profiles import BindingMixtureProfile
from dit.shannon import entropy as H


def make_and():
    outs, p = [], []
    for x in "01":
        for y in "01":
            z = "1" if (x == "1" and y == "1") else "0"
            outs.append(x + y + z)
            p.append(0.25)
    return Distribution(outs, p)


DISTS = {
    "GB": Distribution(["000", "111"], [0.5, 0.5]),
    "XOR": Distribution(["000", "011", "101", "110"], [0.25] * 4),
    "W": Distribution(["001", "010", "100"], [1 / 3] * 3),
    "Copy": Distribution(["000", "001", "110", "111"], [0.25] * 4),
    "AND": make_and(),
}


def cartesian(dist):
    d = prepare_dist(deepcopy(dist))
    n = d.outcome_length()
    alph = [tuple(sorted({o[i] for o in d.outcomes})) for i in range(n)]
    outs = list(product(*alph))
    pmf_map = {tuple(o): float(p) for o, p in zip(d.outcomes, d.pmf, strict=True)}
    pmf = np.array([pmf_map.get(o, 0.0) for o in outs], dtype=float)
    pmf /= pmf.sum()
    return outs, pmf, alph


def _jsd_pmf(p, q):
    return float(jensen_shannon_divergence_pmf(np.vstack([p, q])))


def _fwd_kl(p, q, eps=1e-300):
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], eps))))


def _rev_kl(q, p, eps=1e-16):
    if np.any((q > eps) & (p <= eps)):
        return float("inf")
    from scipy.special import xlogy

    return float(np.sum(xlogy(q, q) - xlogy(q, np.maximum(p, eps))) / np.log(2))


def metrics(outs, p, q):
    P = Distribution(outs, p, base="linear", validate=False)
    Q = Distribution(outs, q, base="linear", validate=False)
    P.normalize()
    Q.normalize()
    return {
        "B": float(B(Q)),
        "T": float(T(Q)),
        "H": float(H(Q)),
        "Df": _fwd_kl(p, q),
        "Dr": _rev_kl(q, p),
        "JSD": _jsd_pmf(p, q),
        "L2": float(np.linalg.norm(p - q)),
        "BP": float(B(P)),
        "TP": float(T(P)),
    }


def project_amari(outs, pmf, alph, order, criterion, eps=1e-6, nrestarts=10, maxiter=2500):
    """Project onto M_order under fwd KL / rev KL / JSD."""
    A, _ = mflat_design_matrix(alph, order)
    n_out, n_feat = A.shape
    rng = np.random.default_rng(0)

    target = pmf.copy()
    if criterion == "rev" and eps:
        dtmp = Distribution(outs, pmf, base="linear", validate=False)
        dtmp = symmetric_smooth(dtmp, eps)
        target = _aligned_pmf(dtmp, outs)

    def feasible(th):
        q = A @ th
        if np.any(q < -1e-10):
            q = np.maximum(q, 0.0)
        s = q.sum()
        if s <= 0:
            return None
        return q / s

    def objective(th):
        q = feasible(th)
        if q is None:
            return 1e6
        if criterion == "fwd":
            return _fwd_kl(pmf, q)
        if criterion == "rev":
            val = _rev_kl(q, target)
            return val if np.isfinite(val) else 1e6
        if criterion == "jsd":
            return _jsd_pmf(pmf, q)
        raise ValueError(criterion)

    cons = [{"type": "eq", "fun": lambda th: float((A @ th).sum() - 1.0)}]
    for i in range(n_out):
        cons.append({"type": "ineq", "fun": lambda th, i=i: float((A @ th)[i])})

    initials = [np.linalg.lstsq(A, pmf, rcond=None)[0], np.zeros(n_feat)]
    initials[1][0] = 1.0 / n_out
    for _ in range(max(0, nrestarts - len(initials))):
        initials.append(rng.normal(0.0, 0.05, size=n_feat))

    best_q, best_val = None, np.inf
    for th0 in initials:
        q0 = A @ th0
        if abs(q0.sum()) > 0:
            th0 = th0 / q0.sum()
        res = minimize(
            objective,
            th0,
            method="SLSQP",
            constraints=cons,
            options={"maxiter": maxiter, "ftol": 1e-14, "disp": False},
        )
        q = feasible(res.x)
        if q is None:
            continue
        val = objective(res.x)
        if val < best_val:
            best_val = val
            best_q = q

    if best_q is None:
        raise RuntimeError(f"Amari projection failed ({criterion}, order={order})")
    return best_q, float(best_val)


def embed_PS(outs, pmf, alph, S, mode):
    n = len(alph)
    marg = defaultdict(float)
    for o, p in zip(outs, pmf, strict=True):
        marg[tuple(o[i] for i in S)] += p
    rest = [i for i in range(n) if i not in S]
    if mode == "uniform":
        rest_size = int(np.prod([len(alph[i]) for i in rest])) if rest else 1
        return np.array([marg[tuple(o[i] for i in S)] / rest_size for o in outs])
    ones = []
    for i in range(n):
        m1 = defaultdict(float)
        for o, p in zip(outs, pmf, strict=True):
            m1[o[i]] += p
        ones.append(m1)
    q = []
    for o in outs:
        val = marg[tuple(o[i] for i in S)]
        for i in rest:
            val *= ones[i][o[i]]
        q.append(val)
    q = np.asarray(q, dtype=float)
    return q / q.sum()


def fit_option2(outs, pmf, alph, order, mode, n_init=12):
    blocks, labels = [], []
    blocks.append(np.ones(len(outs)) / len(outs))
    labels.append("∅")
    for k in range(1, order + 1):
        for S in combinations(range(len(alph)), k):
            blocks.append(embed_PS(outs, pmf, alph, S, mode))
            labels.append("".join(map(str, S)))
    A = np.column_stack(blocks)
    nb = A.shape[1]

    def loss(x):
        return np.sum((A @ x - pmf) ** 2)

    cons = {"type": "eq", "fun": lambda x: x.sum() - 1}
    best = None
    rng = np.random.default_rng(0)
    for _ in range(n_init):
        x0 = rng.dirichlet(np.ones(nb))
        r = minimize(
            loss,
            x0,
            bounds=[(0, None)] * nb,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    alpha = np.maximum(best[1], 0.0)
    alpha /= alpha.sum()
    q = A @ alpha
    q = np.maximum(q, 0.0)
    q /= q.sum()
    return labels, alpha, q, float(np.sqrt(best[0]))


def fmt_alpha(labels, alpha, thresh=0.05):
    parts = [f"{l}:{a:.2f}" for l, a in sorted(zip(labels, alpha, strict=True), key=lambda t: -t[1]) if a >= thresh]
    return "[" + ", ".join(parts) + "]"


def main():
    print("=" * 88)
    print("LINEAR BUILDING-BLOCK PROBE")
    print("=" * 88)

    for name, dist in DISTS.items():
        outs, pmf, alph = cartesian(dist)
        n = len(alph)
        print(f"\n##### {name}  |supp|={int((pmf > 0).sum())}  B={B(dist):.4f} T={T(dist):.4f} H={H(dist):.4f}")

        print("\n-- (1) Amari M_k --")
        print(f"{'k':>2} {'crit':>4} {'obj':>8} {'B(Q)':>8} {'T(Q)':>8} {'Df':>8} {'Dr':>8} {'JSD':>8} {'L2':>8}")
        for order in range(1, n + 1):
            for crit in ("fwd", "rev", "jsd"):
                q, obj = project_amari(outs, pmf, alph, order, crit)
                m = metrics(outs, pmf, q)
                print(
                    f"{order:2d} {crit:>4} {obj:8.4f} {m['B']:8.4f} {m['T']:8.4f} "
                    f"{m['Df']:8.4f} {m['Dr']:8.4f} {m['JSD']:8.4f} {m['L2']:8.4f}"
                )

        print("\n-- (2) Convex combo of Lift(P_S) --")
        print(f"{'k':>2} {'lift':>7} {'B(Q)':>8} {'T(Q)':>8} {'Df':>8} {'JSD':>8} {'L2':>8}  alpha")
        for order in range(1, n + 1):
            for mode in ("uniform", "product"):
                labels, alpha, q, l2 = fit_option2(outs, pmf, alph, order, mode)
                m = metrics(outs, pmf, q)
                print(
                    f"{order:2d} {mode:>7} {m['B']:8.4f} {m['T']:8.4f} {m['Df']:8.4f} "
                    f"{m['JSD']:8.4f} {m['L2']:8.4f}  {fmt_alpha(labels, alpha)}"
                )

        print("\n-- Contrast: BindingMixtureProfile ΔB --")
        bmp = BindingMixtureProfile(dist, k_max=min(8, max(4, int((pmf > 0).sum()))), n_init=8, seed=0)
        for k in sorted(bmp.profile):
            print(
                f"  k={k}: ΔB={bmp.profile[k]:.4f}  B(Q)={bmp.bindings[k - 1]:.4f}  "
                f"D(P||Q)={bmp.forward_kl[k - 1]:.4f}"
            )


if __name__ == "__main__":
    main()
