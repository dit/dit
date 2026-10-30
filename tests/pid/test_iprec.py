"""
Tests for dit.pid.measures.iprec.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures.iprec import PID_Prec


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
    pytest.importorskip("pypoman")
    d = bivariates["sum"]
    exact = PID_Prec(d, [[0], [1]], [2], compute=False)[((0,), (1,))]
    bounded = PID_Prec(d, [[0], [1]], [2], compute=False, n_q=2)[((0,), (1,))]
    assert bounded <= exact + 1e-9
    assert exact == pytest.approx(0.5, abs=1e-9)
