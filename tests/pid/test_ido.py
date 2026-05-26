"""
Tests for dit.pid.measures.ido (do-calculus PID).
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.ido import PID_Do, _intervened_joint


def test_pid_do_redundant():
    """
    Redundant distribution (X_0 = X_1 = Y): 1 bit redundancy.
    """
    d = bivariates["redundant"]
    pid = PID_Do(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(1.0, abs=1e-9)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-9)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-9)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-9)


def test_pid_do_synergy():
    """
    Synergy / XOR: redundancy = 0, synergy = 1 bit.

    X_0, X_1 independent uniform bits, Y = X_0 XOR X_1.  All marginals
    are uniform and X'_0 is independent of X_1 in the intervened joint,
    so I(X'_0; X_1) = 0.
    """
    d = bivariates["synergy"]
    pid = PID_Do(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-9)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-9)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-9)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-9)


def test_pid_do_unique1():
    """
    Unique-1 (Y = X_1, X_0 independent of Y): redundancy = 0,
    all information unique to X_1.
    """
    d = bivariates["unique 1"]
    pid = PID_Do(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-9)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-9)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-9)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-9)


def test_pid_do_cat():
    """
    cat (Y = (X_0, X_1) concatenation): independent sources, deterministic
    target, so each source has 1 bit unique and redundancy = 0.
    """
    d = bivariates["cat"]
    pid = PID_Do(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-9)
    assert pid[((0,),)] == pytest.approx(1.0, abs=1e-9)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-9)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-9)


def test_pid_do_sum():
    """
    sum (Y = X_0 + X_1): closed-form check.

    The intervened joint over (X'_0, X_1) is
        [[3/8, 1/8],
         [1/8, 3/8]]
    so I(X'_0; X_1) = 2 * (3/8 * log2(3/2) + 1/8 * log2(1/2)) ~= 0.1887 bits.
    """
    import math

    d = bivariates["sum"]
    pid = PID_Do(d, ((0,), (1,)), (2,))
    expected_red = 2 * (0.375 * math.log2(1.5) + 0.125 * math.log2(0.5))
    assert pid[((0,), (1,))] == pytest.approx(expected_red, abs=1e-9)


def test_pid_do_symmetry():
    """
    The two algebraically-equivalent forms I(X'_0; X_1) and I(X'_1; X_0)
    agree to machine precision on the redundant distribution.
    """
    d = bivariates["redundant"].coalesce([(0,), (1,), (2,)])
    d.make_dense()
    p_xyz = d.pmf.reshape([len(a) for a in d.alphabet])

    q_for_0 = _intervened_joint(p_xyz)
    q_for_1 = _intervened_joint(p_xyz.transpose(1, 0, 2))

    def _mi(q_ab):
        import numpy as np

        p_a = q_ab.sum(axis=1)
        p_b = q_ab.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = p_a[:, None] * p_b[None, :]
            ratio = np.where((q_ab > 0) & (denom > 0), q_ab / denom, 1.0)
        return float(np.nansum(q_ab * np.log2(ratio)))

    mi_a = _mi(q_for_0.sum(axis=2))
    mi_b = _mi(q_for_1.sum(axis=2))
    assert mi_a == pytest.approx(mi_b, abs=1e-12)


def test_pid_do_only_bivariate():
    """
    I_do is bivariate-only; computing it on three sources should raise
    via the class-level guard in _measure.
    """
    from dit import Distribution
    from dit.exceptions import ditException

    outcomes = ["0000", "1110", "1101", "1011"]
    pmf = [1 / 4] * 4
    d = Distribution(outcomes, pmf)
    with pytest.raises(ditException):
        PID_Do._measure(d, ((0,), (1,), (2,)), (3,))
