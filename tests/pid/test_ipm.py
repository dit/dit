"""
Tests for dit.pid.ipm.
"""

import numpy as np
import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.ipm import PID_PM


def test_pid_pm1():
    """
    Test ipm on a generic distribution.
    """
    d = bivariates["pnt. unq."]
    pid = PID_PM(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0)
    assert pid[((0,),)] == pytest.approx(0.5)
    assert pid[((1,),)] == pytest.approx(0.5)
    assert pid[((0, 1),)] == pytest.approx(0.0)


def test_pid_pm2():
    """
    Test imin on another generic distribution.
    """
    d = bivariates["unique 1"]
    pid = PID_PM(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(1.0)
    assert pid[((0,),)] == pytest.approx(-1.0)
    assert pid[((1,),)] == pytest.approx(0.0)
    assert pid[((0, 1),)] == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Pointwise tests
# -----------------------------------------------------------------------


def test_pid_pm_pointwise_produces_dicts():
    """pointwise=True populates per-outcome dicts at every lattice node."""
    d = bivariates["pnt. unq."]
    pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

    for node in pid._lattice:
        pw_red = pid.get_pw_red(node)
        pw_pi = pid.get_pw_pi(node)
        assert isinstance(pw_red, dict)
        assert isinstance(pw_pi, dict)
        assert set(pw_red.keys()) == set(d.outcomes)
        assert set(pw_pi.keys()) == set(d.outcomes)


def test_pid_pm_pointwise_average_matches_scalar():
    """Averaging pointwise PI atoms reproduces the standard PID values."""
    for name in ["pnt. unq.", "synergy", "redundant", "unique 1", "reduced or"]:
        d = bivariates[name]
        pid_scalar = PID_PM(d, ((0,), (1,)), (2,))
        pid_pw = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

        for node in pid_scalar._lattice:
            avg_red = np.nansum([d[o] * pid_pw.get_pw_red(node)[o] for o in d.outcomes])
            avg_pi = np.nansum([d[o] * pid_pw.get_pw_pi(node)[o] for o in d.outcomes])
            assert avg_red == pytest.approx(pid_scalar.get_red(node), abs=1e-10)
            assert avg_pi == pytest.approx(pid_scalar.get_pi(node), abs=1e-10)


def test_pid_pm_plus_minus_parts():
    """
    Verify that plus/minus parts are returned and that
    pw_red = pw_red_plus - pw_red_minus for every outcome.
    """
    d = bivariates["pnt. unq."]
    pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

    for node in pid._lattice:
        for o in d.outcomes:
            diff = pid.get_pw_red_plus(node)[o] - pid.get_pw_red_minus(node)[o]
            expected = pid.get_pw_red(node)[o]
            if np.isnan(diff) and np.isnan(expected):
                continue
            assert diff == pytest.approx(expected, abs=1e-10)


def test_pid_pm_specificity_nonneg():
    """
    The specificity (plus) component min_h_s is always non-negative since
    it is the minimum of surprisals.
    """
    for name in ["pnt. unq.", "synergy", "redundant", "unique 1", "reduced or", "and", "sum"]:
        d = bivariates[name]
        pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

        for node in pid._lattice:
            for o in d.outcomes:
                assert pid.get_pw_red_plus(node)[o] >= -1e-10, f"specificity negative at {name}, {node}, {o}"


def test_pid_pm_ambiguity_nonneg():
    """
    The ambiguity (minus) component min_h_s|t is always non-negative since
    it is the minimum of conditional surprisals.
    """
    for name in ["pnt. unq.", "synergy", "redundant", "unique 1", "reduced or", "and", "sum"]:
        d = bivariates[name]
        pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

        for node in pid._lattice:
            for o in d.outcomes:
                assert pid.get_pw_red_minus(node)[o] >= -1e-10, f"ambiguity negative at {name}, {node}, {o}"


def test_pid_pm_pnt_unq_pointwise_values():
    """
    Pointwise unique distribution: verify pointwise unique atoms show the
    expected per-outcome structure (matching PID_SX behaviour).
    """
    d = bivariates["pnt. unq."]
    pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

    unq_0 = ((0,),)
    unq_1 = ((1,),)

    assert pid.get_pw_pi(unq_0)["101"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_0)["202"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_0)["011"] == pytest.approx(0.0, abs=1e-10)
    assert pid.get_pw_pi(unq_0)["022"] == pytest.approx(0.0, abs=1e-10)

    assert pid.get_pw_pi(unq_1)["011"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_1)["022"] == pytest.approx(1.0, abs=1e-10)
    assert pid.get_pw_pi(unq_1)["101"] == pytest.approx(0.0, abs=1e-10)
    assert pid.get_pw_pi(unq_1)["202"] == pytest.approx(0.0, abs=1e-10)


def test_pid_pm_xor_synergy():
    """
    XOR: all information is synergistic.  Pointwise synergy atom should be
    positive for every outcome and the redundancy atom should be zero.
    """
    d = bivariates["synergy"]
    pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

    syn_node = ((0, 1),)
    rdn_node = ((0,), (1,))

    for o in d.outcomes:
        assert pid.get_pw_pi(syn_node)[o] == pytest.approx(1.0, abs=1e-10)
        assert pid.get_pw_pi(rdn_node)[o] == pytest.approx(0.0, abs=1e-10)


def test_pid_pm_redundant_distribution():
    """
    Redundant distribution (both sources copy target): all information
    should be in the redundancy atom.
    """
    d = bivariates["redundant"]
    pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)

    rdn_node = ((0,), (1,))
    for o in d.outcomes:
        assert pid.get_pw_pi(rdn_node)[o] == pytest.approx(pid.get_pw_red(rdn_node)[o], abs=1e-10)


def test_pid_pm_pw_to_string():
    """pw_to_string returns a non-empty string when pointwise=True."""
    d = bivariates["pnt. unq."]
    pid = PID_PM(d, ((0,), (1,)), (2,), pointwise=True)
    s = pid.pw_to_string()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "i_r" in s
