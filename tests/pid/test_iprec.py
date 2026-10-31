"""
Tests for dit.pid.measures.iprec.
"""

from fractions import Fraction

import numpy as np
import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures.ideg import _double_description, _integer_row
from dit.pid.measures.iprec import PID_Prec

# On this distribution, vertex enumeration in floating point (pypoman) raised "Numerical
# inconsistency is found" for n_q = 5, the cardinality from Theorem A1, and for n_q = 4
# once the probabilities were rounded differently.
ILL_CONDITIONED = Distribution(
    [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0), (1, 1, 0), (2, 0, 0), (2, 0, 1), (2, 2, 0), (2, 2, 1)],
    np.array([159, 212, 217, 154, 1, 104, 111, 16, 27]) / 1001,
)


def _brute_force(d, n_q):
    """
    The largest I(Q; Y) over the vertices of the polytope of degradation channels of the
    sources X0, X1 with n_q outcomes, enumerated exactly as the extreme rays of the
    homogenized cone.
    """
    pjoint = d.coalesce([[0], [1], [2]])
    xs, ys = pjoint.alphabet[:2], pjoint.alphabet[2]
    pxy = [{(x, y): Fraction(0) for x in xs[i] for y in ys} for i in range(2)]
    for outcome, p in zip(pjoint.outcomes, pjoint.pmf, strict=True):
        for i in range(2):
            pxy[i][outcome[i], outcome[2]] += Fraction(p)
    index = {(i, x, q): k for k, (i, x, q) in enumerate((i, x, q) for i in range(2) for x in xs[i] for q in range(n_q))}
    dim = len(index) + 1  # with the homogenizing coordinate last
    equalities = []
    for i in range(2):
        for x in xs[i]:
            row = [0] * dim
            for q in range(n_q):
                row[index[i, x, q]] = 1
            row[-1] = -1
            equalities.append(row)
    for q in range(n_q):
        for y in ys:
            row = [Fraction(0)] * dim
            for x in xs[0]:
                row[index[0, x, q]] += pxy[0][x, y]
            for x in xs[1]:
                row[index[1, x, q]] -= pxy[1][x, y]
            equalities.append(_integer_row(row))
    p0 = np.array([[float(pxy[0][x, y]) for y in ys] for x in xs[0]])
    columns = [[index[0, x, q] for q in range(n_q)] for x in xs[0]]
    best = 0.0
    for ray in _double_description(equalities, dim):
        p_qy = (ray[columns] / ray[-1]).T @ p0
        outer = np.outer(p_qy.sum(axis=1), p_qy.sum(axis=0))
        mask = p_qy > 0
        best = max(best, float((p_qy[mask] * np.log2(p_qy[mask] / outer[mask])).sum()))
    return best


def test_pid_prec1():
    """
    Test iprec on a generic distribution.
    """
    d = bivariates["and"]
    pid = PID_Prec(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.31127812445913294, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-3)


def test_pid_prec2():
    """
    Test iprec on another generic distribution.
    """
    d = bivariates["sum"]
    pid = PID_Prec(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


def test_pid_prec3():
    """
    Test iprec on a generic trivariate source distribution.
    """
    events = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
    d = Distribution(events, [1 / 8] * 8)
    pid = PID_Prec(d, [[0], [1], [2]], [3], compute=False)
    assert pid[((0,), (1,), (2,))] == pytest.approx(0.13795718192252743, abs=1e-3)


def test_pid_prec4():
    """
    Test iprec on unique gate X1,X2 -> X1, which should have redundancy equal to I(X1;X2)
    """
    d = Distribution(["000", "011", "100", "111"], [0.35, 0.15, 0.15, 0.35])
    pid = PID_Prec(
        d,
        [
            [0],
            [1],
        ],
        [2],
        compute=False,
    )
    assert pid[
        (
            (0,),
            (1,),
        )
    ] == pytest.approx(0.119, abs=1e-2)


def test_pid_prec_trivariate_full():
    """
    The full trivariate decomposition of test_pid_prec3, which vertex enumeration
    could not compute in reasonable time. Vertex enumeration with n_q = 4 gives the
    same bottom redundancy, to 1e-10.
    """
    events = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
    d = Distribution(events, [1 / 8] * 8)
    pid = PID_Prec(d, [[0], [1], [2]], [3])
    assert pid[((0,), (1,), (2,))] == pytest.approx(0.13792538097, abs=1e-9)
    assert pid.complete
    assert pid.consistent


@pytest.mark.parametrize("name", ["boom", "sum", "pnt. unq."])
def test_pid_prec_solution(name):
    """The optimal Q is a common degradation of the sources attaining the redundancy."""
    d = bivariates[name]
    value, sol = PID_Prec.get_solution_information(d, ((0,), (1,)), (2,))
    p_qy = sol["p(Q,Y)"]
    p_q, p_y = p_qy.sum(axis=1), p_qy.sum(axis=0)
    nz = p_qy > 0
    mi = np.sum(p_qy[nz] * np.log2(p_qy[nz] / np.outer(p_q, p_y)[nz]))
    assert p_qy.min() >= -1e-12
    assert mi == pytest.approx(value, abs=1e-9)

    pjoint = d.coalesce([[0], [1], [2]])
    for i in range(2):
        joint = pjoint.marginal([i, 2]).to_dict()
        p_xy = np.array([[joint.get((x, y), 0.0) for y in pjoint.alphabet[2]] for x in pjoint.alphabet[i]])
        assert np.allclose(sol[f"p(Q|X{i})"].sum(axis=0), 1)
        assert np.allclose(sol[f"p(Q|X{i})"] @ p_xy, p_qy)


def test_pid_prec_fixed_cardinality():
    """With n_q fixed, vertex enumeration gives a lower bound on the exact value."""
    d = bivariates["sum"]
    exact = PID_Prec(d, [[0], [1]], [2], compute=False)[((0,), (1,))]
    bounded = PID_Prec(d, [[0], [1]], [2], compute=False, n_q=2)[((0,), (1,))]
    assert bounded <= exact + 1e-9
    assert exact == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize(("n_q", "value"), [(2, 0.0366386), (3, 0.0371232), (4, 0.0371232), (5, 0.0371232)])
def test_pid_prec_fixed_cardinality_ill_conditioned(n_q, value):
    """
    The value for a fixed cardinality is found exactly, where floating-point vertex
    enumeration failed.
    """
    assert PID_Prec._measure(ILL_CONDITIONED, [[0], [1]], [2], n_q=n_q) == pytest.approx(value, abs=1e-7)


@pytest.mark.parametrize(
    ("d", "n_q"),
    [(bivariates["and"], 2), (bivariates["sum"], 2), (ILL_CONDITIONED, 2), (ILL_CONDITIONED, 3)],
)
def test_pid_prec_fixed_cardinality_brute_force(d, n_q):
    """The value for a fixed cardinality is the maximum over all vertices of the polytope."""
    value, _ = PID_Prec.get_solution_information(d, [[0], [1]], [2], n_q=n_q)
    assert value == pytest.approx(_brute_force(d, n_q), abs=1e-9)


def test_pid_prec_fixed_cardinality_solution():
    """
    The value grows with the cardinality up to the exact value at the cardinality of
    Theorem A1, and the solution is a common degradation which attains it.
    """
    exact = PID_Prec._measure(ILL_CONDITIONED, [[0], [1]], [2])
    pjoint = ILL_CONDITIONED.coalesce([[0], [1], [2]])
    values = []
    for n_q in range(2, 6):
        value, sol = PID_Prec.get_solution_information(ILL_CONDITIONED, [[0], [1]], [2], n_q=n_q)
        values.append(value)
        p_qy = sol["p(Q,Y)"]
        assert p_qy.shape == (n_q, 2)
        for i in range(2):
            joint = pjoint.marginal([i, 2]).to_dict()
            p_xy = np.array([[joint.get((x, y), 0.0) for y in pjoint.alphabet[2]] for x in pjoint.alphabet[i]])
            assert sol[f"p(Q|X{i})"].shape == (n_q, 3)
            assert sol[f"p(Q|X{i})"].min() >= 0
            assert np.allclose(sol[f"p(Q|X{i})"].sum(axis=0), 1)
            assert np.allclose(sol[f"p(Q|X{i})"] @ p_xy, p_qy)
        mask = p_qy > 0
        outer = np.outer(p_qy.sum(axis=1), p_qy.sum(axis=0))
        assert (p_qy[mask] * np.log2(p_qy[mask] / outer[mask])).sum() == pytest.approx(value, abs=1e-9)
    assert np.all(np.diff(values) >= -1e-12)
    assert values[0] < exact - 1e-4
    assert values[-1] == pytest.approx(exact, abs=1e-9)
