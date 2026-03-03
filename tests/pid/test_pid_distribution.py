"""
Tests for PID measures with Distribution inputs.

Each test converts a bivariate distribution to Distribution, computes the
PID, and verifies the results match the Distribution-based computation.
"""

import pytest

from dit.distribution import Distribution
from dit.pid.distributions import bivariates


def _to_xr(d):
    """Convert a Distribution to an Distribution with standard dim names."""
    n = d.outcome_length()
    names = [f"X{i}" for i in range(n)]
    return Distribution.from_distribution(d, names)


# ───────────────────────────────────────────────────────────────────────
# PID_RDR (irdr) -- fast
# ───────────────────────────────────────────────────────────────────────


class TestPID_RDR_Dist:
    def test_diff(self):
        from dit.pid.measures.irdr import PID_RDR

        d = bivariates["diff"]
        pid_d = PID_RDR(d, ((0,), (1,)), (2,))
        pid_x = PID_RDR(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)

    def test_reduced_or(self):
        from dit.pid.measures.irdr import PID_RDR

        d = bivariates["reduced or"]
        pid_d = PID_RDR(d)
        pid_x = PID_RDR(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)

    def test_properties(self):
        from dit.pid.measures.irdr import PID_RDR

        d = bivariates["and"]
        pid = PID_RDR(_to_xr(d))
        assert pid.complete
        assert pid.nonnegative
        assert pid.consistent


# ───────────────────────────────────────────────────────────────────────
# PID_Proj (iproj) -- fast for bivariate
# ───────────────────────────────────────────────────────────────────────


class TestPID_Proj_Dist:
    def test_and(self):
        from dit.pid.measures.iproj import PID_Proj

        d = bivariates["and"]
        pid_d = PID_Proj(d, ((0,), (1,)), (2,))
        pid_x = PID_Proj(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)

    def test_properties(self):
        from dit.pid.measures.iproj import PID_Proj

        d = bivariates["synergy"]
        pid = PID_Proj(_to_xr(d))
        assert pid.complete
        assert pid.nonnegative
        assert pid.consistent


# ───────────────────────────────────────────────────────────────────────
# PID_Prec (iprec) -- moderate speed
# ───────────────────────────────────────────────────────────────────────


class TestPID_Prec_Dist:
    def test_and(self):
        from dit.pid.measures.iprec import PID_Prec

        d = bivariates["and"]
        pid_d = PID_Prec(d, ((0,), (1,)), (2,))
        pid_x = PID_Prec(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)

    def test_properties(self):
        from dit.pid.measures.iprec import PID_Prec

        d = bivariates["synergy"]
        pid = PID_Prec(_to_xr(d))
        assert pid.complete
        assert pid.nonnegative
        assert pid.consistent


# ───────────────────────────────────────────────────────────────────────
# PID_PM (ipm) -- fast
# ───────────────────────────────────────────────────────────────────────


class TestPID_PM_Dist:
    def test_and(self):
        from dit.pid.measures.ipm import PID_PM

        d = bivariates["and"]
        pid_d = PID_PM(d, ((0,), (1,)), (2,))
        pid_x = PID_PM(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)

    def test_synergy(self):
        from dit.pid.measures.ipm import PID_PM

        d = bivariates["synergy"]
        pid_d = PID_PM(d, ((0,), (1,)), (2,))
        pid_x = PID_PM(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)


# ───────────────────────────────────────────────────────────────────────
# PID_SX (isx) -- fast
# ───────────────────────────────────────────────────────────────────────


class TestPID_SX_Dist:
    def test_and(self):
        from dit.pid.measures.isx import PID_SX

        d = bivariates["and"]
        pid_d = PID_SX(d, ((0,), (1,)), (2,))
        pid_x = PID_SX(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)

    def test_synergy(self):
        from dit.pid.measures.isx import PID_SX

        d = bivariates["synergy"]
        pid_d = PID_SX(d, ((0,), (1,)), (2,))
        pid_x = PID_SX(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)


# ───────────────────────────────────────────────────────────────────────
# PID_WB (imin) -- slow, so only test properties on tiny dist
# ───────────────────────────────────────────────────────────────────────


class TestPID_WB_Dist:
    def test_rdn_properties(self):
        """Use the redundant distribution (trivially fast)."""
        from dit.pid.measures.imin import PID_WB

        d = bivariates["redundant"]
        pid = PID_WB(_to_xr(d))
        assert pid.complete
        assert pid.nonnegative
        assert pid.consistent

    def test_rdn_values(self):
        from dit.pid.measures.imin import PID_WB

        d = bivariates["redundant"]
        pid_d = PID_WB(d, ((0,), (1,)), (2,))
        pid_x = PID_WB(_to_xr(d))
        for node in pid_d._lattice:
            assert pid_x.get_pi(node) == pytest.approx(pid_d.get_pi(node), abs=1e-4)
