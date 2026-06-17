"""
Exponential cone programming for BROJA bivariate PID.

Implements the EXP formulation of Makkeh, Theis, and Vicente
(:cite:`makkeh2018broja`, arXiv:1802.02485).

The algorithm follows the BROJA_2PID reference implementation (Apache 2.0).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import sparse

from ..distribution import Distribution
from .pid_broja import prepare_dist as broja_prepare_dist

__all__ = (
    "broja_cone_dist",
    "broja_cone_solve",
)

_LN2 = math.log(2)


def _r_idx(i):
    return 3 * i


def _p_idx(i):
    return 3 * i + 1


def _q_idx(i):
    return 3 * i + 2


class _BrojaConeSolver:
    """Build and solve the EXP cone program for a sparse (x,y,z) support."""

    def __init__(self, marg_xy, marg_xz):
        self.b_xy = dict(marg_xy)
        self.b_xz = dict(marg_xz)
        self.X = sorted({x for x, _ in self.b_xy} | {x for x, _ in self.b_xz})
        self.Y = sorted({y for _, y in self.b_xy})
        self.Z = sorted({z for _, z in self.b_xz})
        self.idx_of_trip = {}
        self.trip_of_idx = []
        for x in self.X:
            for y in self.Y:
                if (x, y) not in self.b_xy:
                    continue
                for z in self.Z:
                    if (x, z) not in self.b_xz:
                        continue
                    self.idx_of_trip[(x, y, z)] = len(self.trip_of_idx)
                    self.trip_of_idx.append((x, y, z))

        self.c = None
        self.G = None
        self.h = None
        self.dims = {}
        self.A = None
        self.b = None
        self.sol_rpq = None
        self.sol_lambda = None

    def create_model(self):
        n = len(self.trip_of_idx)
        m = len(self.b_xy) + len(self.b_xz)
        n_vars = 3 * n
        n_cons = n + m

        self.b_vec = np.zeros(n_cons, dtype=float)
        eqn, var, coeff = [], [], []

        for i, (_x, y, z) in enumerate(self.trip_of_idx):
            eqn.append(i)
            var.append(_p_idx(i))
            coeff.append(-1.0)
            for u in self.X:
                key = (u, y, z)
                if key in self.idx_of_trip:
                    j = self.idx_of_trip[key]
                    eqn.append(i)
                    var.append(_q_idx(j))
                    coeff.append(1.0)

        eqn_base = len(self.trip_of_idx) - 1

        for x in self.X:
            for y in self.Y:
                if (x, y) not in self.b_xy:
                    continue
                eqn_base += 1
                for z in self.Z:
                    if (x, y, z) in self.idx_of_trip:
                        j = self.idx_of_trip[(x, y, z)]
                        eqn.append(eqn_base)
                        var.append(_q_idx(j))
                        coeff.append(1.0)
                self.b_vec[eqn_base] = self.b_xy[(x, y)]

        for x in self.X:
            for z in self.Z:
                if (x, z) not in self.b_xz:
                    continue
                eqn_base += 1
                for y in self.Y:
                    if (x, y, z) in self.idx_of_trip:
                        j = self.idx_of_trip[(x, y, z)]
                        eqn.append(eqn_base)
                        var.append(_q_idx(j))
                        coeff.append(1.0)
                self.b_vec[eqn_base] = self.b_xz[(x, z)]

        self.A = sparse.csc_matrix((coeff, (eqn, var)), shape=(n_cons, n_vars), dtype=float)

        ieq, var2, coeff2 = [], [], []
        for i in range(n):
            for v in (_r_idx(i), _p_idx(i), _q_idx(i)):
                ieq.append(len(ieq))
                var2.append(v)
                coeff2.append(-1.0)

        self.G = sparse.csc_matrix((coeff2, (ieq, var2)), shape=(n_vars, n_vars), dtype=float)
        self.h = np.zeros(n_vars, dtype=float)
        self.dims = {"e": n}

        self.c = np.zeros(n_vars, dtype=float)
        for i in range(n):
            self.c[_r_idx(i)] = -1.0

    def solve(self, **ecos_kwargs):
        try:
            import ecos
        except ImportError as exc:
            msg = "method='cone' requires the optional 'ecos' package. Install with: pip install dit[broja]"
            raise ImportError(msg) from exc

        solution = ecos.solve(self.c, self.G, self.h, self.dims, self.A, self.b_vec, verbose=False, **ecos_kwargs)
        if "x" not in solution:
            return False
        self.sol_rpq = solution["x"]
        self.sol_lambda = solution["y"]
        self.sol_info = solution.get("info", {})
        return True

    def joint_q(self):
        pdf = {}
        for (x, y, z), i in self.idx_of_trip.items():
            q = self.sol_rpq[_q_idx(i)]
            if q > 0:
                pdf[(x, y, z)] = float(q)
        return pdf

    def check_feasibility(self):
        max_q_neg = 0.0
        for i in range(len(self.trip_of_idx)):
            max_q_neg = max(max_q_neg, -self.sol_rpq[_q_idx(i)])

        max_eq_violation = 0.0
        for xy, target in self.b_xy.items():
            x, y = xy
            total = target
            for z in self.Z:
                if (x, y, z) in self.idx_of_trip:
                    j = self.idx_of_trip[(x, y, z)]
                    total -= max(0.0, self.sol_rpq[_q_idx(j)])
            max_eq_violation = max(max_eq_violation, abs(total))

        for xz, target in self.b_xz.items():
            x, z = xz
            total = target
            for y in self.Y:
                if (x, y, z) in self.idx_of_trip:
                    j = self.idx_of_trip[(x, y, z)]
                    total -= max(0.0, self.sol_rpq[_q_idx(j)])
            max_eq_violation = max(max_eq_violation, abs(total))

        primal_infeas = max(max_q_neg, max_eq_violation)

        idx_xy = {(x, y): i for i, (x, y) in enumerate(self.b_xy)}
        idx_xz = {(x, z): i for i, (x, z) in enumerate(self.b_xz)}

        mu_yz = defaultdict(float)
        for j, (_x, y, z) in enumerate(self.trip_of_idx):
            mu_yz[(y, z)] += self.sol_lambda[j]

        dual_infeas = 0.0
        for i, (x, y, z) in enumerate(self.trip_of_idx):
            xy_idx = len(self.trip_of_idx) + idx_xy[(x, y)]
            xz_idx = len(self.trip_of_idx) + len(self.b_xy) + idx_xz[(x, z)]
            dual_infeas = max(
                dual_infeas,
                -self.sol_lambda[xy_idx] - self.sol_lambda[xz_idx] - mu_yz[(y, z)] - math.log(-self.sol_lambda[i]) - 1,
            )

        condent = self._condentropy()
        dual_val = -float(np.dot(self.sol_lambda, self.b_vec))
        gap = max(-condent * _LN2 - dual_val, 0.0)
        return primal_infeas, dual_infeas, gap

    def _condentropy(self):
        total = 0.0
        for y in self.Y:
            for z in self.Z:
                q_list = [_q_idx(self.idx_of_trip[(x, y, z)]) for x in self.X if (x, y, z) in self.idx_of_trip]
                marg = sum(max(0.0, self.sol_rpq[i]) for i in q_list)
                if marg <= 0:
                    continue
                for i in q_list:
                    q = self.sol_rpq[i]
                    if q > 0:
                        total -= q * math.log(q / marg)
        return total

    def cond_y_mutinf(self):
        marg_yz = self._marg_yz()
        marg_y = defaultdict(float)
        for (y, _z), p in marg_yz.items():
            marg_y[y] += p

        total = 0.0
        for x in self.X:
            for z in self.Z:
                if (x, z) not in self.b_xz:
                    continue
                for y in self.Y:
                    if (x, y, z) not in self.idx_of_trip:
                        continue
                    i = _q_idx(self.idx_of_trip[(x, y, z)])
                    q = self.sol_rpq[i]
                    yz = marg_yz.get((y, z), 0.0)
                    if q > 0 and yz > 0 and self.b_xy[(x, y)] > 0:
                        total += q * math.log(q * marg_y[y] / (self.b_xy[(x, y)] * yz))
        return total

    def cond_z_mutinf(self):
        marg_yz = self._marg_yz()
        marg_z = defaultdict(float)
        for (_y, z), p in marg_yz.items():
            marg_z[z] += p

        total = 0.0
        for x in self.X:
            for y in self.Y:
                if (x, y) not in self.b_xy:
                    continue
                for z in self.Z:
                    if (x, y, z) not in self.idx_of_trip:
                        continue
                    i = _q_idx(self.idx_of_trip[(x, y, z)])
                    q = self.sol_rpq[i]
                    yz = marg_yz.get((y, z), 0.0)
                    if q > 0 and yz > 0 and self.b_xz[(x, z)] > 0:
                        total += q * math.log(q * marg_z[z] / (self.b_xz[(x, z)] * yz))
        return total

    def _marg_yz(self):
        if hasattr(self, "_cached_marg_yz"):
            return self._cached_marg_yz
        marg_yz = {}
        for y in self.Y:
            for z in self.Z:
                total = 0.0
                for x in self.X:
                    if (x, y, z) in self.idx_of_trip:
                        q = self.sol_rpq[_q_idx(self.idx_of_trip[(x, y, z)])]
                        if q > 0:
                            total += q
                if total > 0:
                    marg_yz[(y, z)] = total
        self._cached_marg_yz = marg_yz
        return marg_yz


def _marginals_for_cone(d: Distribution):
    """
    Marginals for EXP program.

    BROJA_2PID uses X=target, Y=source0, Z=source1.
    dit coalesced layout is (source0, source1, target) = (Y, Z, X).
    """
    n_y = len(d.alphabet[0])
    n_z = len(d.alphabet[1])
    n_x = len(d.alphabet[2])
    joint = d.pmf.reshape(n_y, n_z, n_x)

    marg_xy = {}
    marg_xz = {}
    for y in range(n_y):
        for x in range(n_x):
            p_xy = float(joint[y, :, x].sum())
            if p_xy > 0:
                marg_xy[(x, y)] = p_xy
    for z in range(n_z):
        for x in range(n_x):
            p_xz = float(joint[:, z, x].sum())
            if p_xz > 0:
                marg_xz[(x, z)] = p_xz
    return marg_xy, marg_xz, (n_y, n_z, n_x)


def broja_cone_solve(dist, sources, target, **ecos_kwargs):
    """
    Solve BROJA bivariate PID via exponential cone programming.

    Returns
    -------
    result : dict
        ``q_dist``, ``converged``, ``primal_infeas``, ``dual_infeas``, ``gap``,
        ``ui_source0``, ``ui_source1`` (in bits).
    """
    d = broja_prepare_dist(dist, sources, target)
    marg_xy, marg_xz, shape = _marginals_for_cone(d)

    solver = _BrojaConeSolver(marg_xy, marg_xz)
    if not solver.trip_of_idx:
        q_dist = d.copy()
        meta = {
            "converged": True,
            "primal_infeas": 0.0,
            "dual_infeas": 0.0,
            "gap": 0.0,
            "ui_source0": 0.0,
            "ui_source1": 0.0,
        }
        return q_dist, meta

    solver.create_model()
    ok = solver.solve(**ecos_kwargs)
    if not ok:
        msg = "ECOS failed to find a solution for the BROJA cone program."
        raise RuntimeError(msg)

    primal_infeas, dual_infeas, gap = solver.check_feasibility()
    converged = primal_infeas < 1e-5

    n_y, n_z, n_x = shape
    pmf = np.zeros(n_y * n_z * n_x, dtype=float)
    pdf = solver.joint_q()
    for (x, y, z), p in pdf.items():
        pmf[y * n_z * n_x + z * n_x + x] = p

    q_dist = d.copy()
    q_dist.pmf = pmf

    ui_y = solver.cond_z_mutinf() / _LN2
    ui_z = solver.cond_y_mutinf() / _LN2
    meta = {
        "converged": converged,
        "primal_infeas": primal_infeas,
        "dual_infeas": dual_infeas,
        "gap": gap,
        "ui_source0": ui_y,
        "ui_source1": ui_z,
    }
    return q_dist, meta


def broja_cone_dist(dist, sources, target, **ecos_kwargs):
    """Return optimal joint distribution from cone programming."""
    q_dist, meta = broja_cone_solve(dist, sources, target, **ecos_kwargs)
    return q_dist, meta
