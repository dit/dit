"""
Tests for dit.pid.isx.
"""

import numpy as np
import pytest

from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.isx import PID_SX


def test_pid_sx1():
    """
    Test isx on a generic distribution.
    """
    d = bivariates["pnt. unq."]
    pid = PID_SX(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0)
    assert pid[((0,),)] == pytest.approx(0.5)
    assert pid[((1,),)] == pytest.approx(0.5)
    assert pid[((0, 1),)] == pytest.approx(0.0)


def test_pid_sx2():
    """
    Test isx on another generic distribution.
    """
    d = bivariates["unique 1"]
    pid = PID_SX(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(np.log2(4 / 3))
    assert pid[((0,),)] == pytest.approx(-np.log2(4 / 3))
    assert pid[((1,),)] == pytest.approx(np.log2(3 / 2))
    assert pid[((0, 1),)] == pytest.approx(np.log2(4 / 3))


def test_pid_sx3():
    """
    Test isx on another generic distribution.
    """
    d = bivariates["reduced or"]
    pid = PID_SX(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(1 / 2 * np.log2(4 / 3) - 1 / 2)
    assert pid[((0,),)] == pytest.approx(1 / 2 * np.log2(4 / 3) + 1 / 4 * np.log2(3))
    assert pid[((1,),)] == pytest.approx(1 / 2 * np.log2(4 / 3) + 1 / 4 * np.log2(3))
    assert pid[((0, 1),)] == pytest.approx(1 / 2 * np.log2(9 / 8))


def test_pid_sx_trivariate():
    """
    Test isx on a trivariate distribution.
    """
    d = trivariates["synergy"]
    pid = PID_SX(d, ((0,), (1,), (2,)), (3,))

    atoms = [
        ((0,),),
        ((1,),),
        ((2,),),
        ((0, 1),),
        ((0, 2),),
        ((1, 2),),
        ((0, 1, 2),),
        ((0,), (1,)),
        ((0,), (2,)),
        ((0,), (1, 2)),
        ((1,), (2,)),
        ((1,), (0, 2)),
        ((2,), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 2)),
        ((0, 2), (1, 2)),
        ((0,), (1,), (2,)),
        ((0, 1), (0, 2), (1, 2)),
    ]
    true_values = np.array(
        [
            np.log2(5 / 4),
            np.log2(5 / 4),
            np.log2(5 / 4),
            np.log2(9 / 8),
            np.log2(9 / 8),
            np.log2(9 / 8),
            np.log2(32 / 27),
            np.log2(7 / 8),
            np.log2(7 / 8),
            np.log2(32 / 35),
            np.log2(7 / 8),
            np.log2(32 / 35),
            np.log2(32 / 35),
            np.log2(16 / 15),
            np.log2(16 / 15),
            np.log2(16 / 15),
            np.log2(8 / 7),
            np.log2(875 / 1024),
        ]
    )

    for atom, true_value in zip(atoms, true_values, strict=True):
        assert pid[atom] == pytest.approx(true_value)


# -----------------------------------------------------------------------
# Pointwise tests
# -----------------------------------------------------------------------


def test_pid_sx_pointwise_produces_dicts():
    """pointwise=True populates per-outcome dicts at every lattice node."""
    d = bivariates["pnt. unq."]
    pid = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)

    for node in pid._lattice:
        pw_red = pid.get_pw_red(node)
        pw_pi = pid.get_pw_pi(node)
        assert isinstance(pw_red, dict)
        assert isinstance(pw_pi, dict)
        assert set(pw_red.keys()) == set(d.outcomes)
        assert set(pw_pi.keys()) == set(d.outcomes)


def test_pid_sx_pointwise_average_matches_scalar():
    """Averaging pointwise PI atoms reproduces the standard PID values."""
    for name in ["pnt. unq.", "synergy", "redundant", "unique 1", "reduced or"]:
        d = bivariates[name]
        pid_scalar = PID_SX(d, ((0,), (1,)), (2,))
        pid_pw = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)

        for node in pid_scalar._lattice:
            avg_red = sum(d[o] * pid_pw.get_pw_red(node)[o] for o in d.outcomes)
            avg_pi = sum(d[o] * pid_pw.get_pw_pi(node)[o] for o in d.outcomes)
            assert avg_red == pytest.approx(pid_scalar.get_red(node), abs=1e-10)
            assert avg_pi == pytest.approx(pid_scalar.get_pi(node), abs=1e-10)


def test_pid_sx_plus_minus_parts_nonneg():
    """
    i^sx+ and i^sx- are non-negative for every outcome (Theorem IV.3,
    Makkeh et al. 2021).
    """
    for name in ["pnt. unq.", "synergy", "redundant", "unique 1", "reduced or", "and", "sum"]:
        d = bivariates[name]
        pid = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)

        for node in pid._lattice:
            for o in d.outcomes:
                assert pid.get_pw_red_plus(node)[o] >= -1e-10, f"pw_red_plus negative at {name}, {node}, {o}"
                assert pid.get_pw_red_minus(node)[o] >= -1e-10, f"pw_red_minus negative at {name}, {node}, {o}"


def test_pid_sx_plus_minus_difference():
    """pw_red = pw_red_plus - pw_red_minus for every outcome."""
    d = bivariates["pnt. unq."]
    pid = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)

    for node in pid._lattice:
        for o in d.outcomes:
            diff = pid.get_pw_red_plus(node)[o] - pid.get_pw_red_minus(node)[o]
            assert diff == pytest.approx(pid.get_pw_red(node)[o], abs=1e-10)


def test_pid_sx_xor_negative_pointwise_pi():
    """
    XOR distribution: the pointwise redundancy PI atom at the bottom node
    ((0,),(1,)) is negative for every outcome, while the averaged PI atom is
    also negative. This is a known property of I^sx on XOR (Makkeh et al.,
    Figure 3).
    """
    d = bivariates["synergy"]
    pid = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)

    bottom = ((0,), (1,))
    for o in d.outcomes:
        assert pid.get_pw_pi(bottom)[o] < 0, f"Expected negative pw_pi at bottom node for XOR outcome {o}"

    assert pid.get_pi(bottom) < 0


def test_pid_sx_pnt_unq_pointwise_values():
    """
    Pointwise unique distribution: source 0 uniquely determines outcome
    for outcomes '101' and '202'; source 1 for '011' and '022'.
    Verify pointwise unique atoms reflect this structure.
    """
    d = bivariates["pnt. unq."]
    pid = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)

    unq_0 = ((0,),)
    unq_1 = ((1,),)

    # outcomes '101','202' carry unique info from source 0
    assert pid.get_pw_pi(unq_0)["101"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_0)["202"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_0)["011"] == pytest.approx(0.0, abs=1e-10)
    assert pid.get_pw_pi(unq_0)["022"] == pytest.approx(0.0, abs=1e-10)

    # outcomes '011','022' carry unique info from source 1
    assert pid.get_pw_pi(unq_1)["011"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_1)["022"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_1)["101"] == pytest.approx(0.0, abs=1e-10)
    assert pid.get_pw_pi(unq_1)["202"] == pytest.approx(0.0, abs=1e-10)


def test_pid_sx_pw_to_string():
    """pw_to_string returns a non-empty string when pointwise=True."""
    d = bivariates["pnt. unq."]
    pid = PID_SX(d, ((0,), (1,)), (2,), pointwise=True)
    s = pid.pw_to_string()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "i_r" in s


def test_pid_sx_pw_to_string_not_computed():
    """pw_to_string returns a message when pointwise=False."""
    d = bivariates["pnt. unq."]
    pid = PID_SX(d, ((0,), (1,)), (2,))
    s = pid.pw_to_string()
    assert "not computed" in s


if __name__ == "__main__":
    test_pid_sx3()
