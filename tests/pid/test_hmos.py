"""
Tests for dit.pid.hmos.
"""

import pytest

from dit import Distribution as D
from dit.pid.distributions import bivariates
from dit.pid.hmos import PED_MOS, h_mos


class TestHMos:
    """Tests for the h_mos redundancy functional."""

    def test_redundant(self):
        d = bivariates["redundant"]
        red = h_mos(d, ((0,), (1,)), (2,))
        assert red == pytest.approx(1.0)

    def test_synergy(self):
        d = bivariates["synergy"]
        red = h_mos(d, ((0,), (1,)), (2,))
        assert red == pytest.approx(1.0)

    def test_two_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        red = h_mos(d, ((0,), (1,)))
        assert red == pytest.approx(1.0)

    def test_two_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        red = h_mos(d, ((0,), (1,)))
        assert red == pytest.approx(1.0)

    def test_correlated_bits(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        red = h_mos(d, ((0,), (1,)))
        assert red == pytest.approx(1.0)

    def test_triadic(self):
        triadic = D(["000", "111", "022", "133", "202", "313", "220", "331"], [1 / 8] * 8)
        red = h_mos(triadic, ((0,), (1,), (2,)))
        assert red == pytest.approx(2.0)


class TestPEDMOS:
    """Tests for the PED_MOS decomposition class."""

    def test_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        ped = PED_MOS(d)
        for atom in ped._lattice:
            if atom == ((0,), (1,)):
                assert ped[atom] == pytest.approx(1.0)
            else:
                assert ped[atom] == pytest.approx(0.0)

    def test_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        ped = PED_MOS(d)
        for atom in ped._lattice:
            if atom == ((0,), (1,)) or atom == ((0, 1),):
                assert ped[atom] == pytest.approx(1.0)
            else:
                assert ped[atom] == pytest.approx(0.0)

    def test_correlated_bits(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        ped = PED_MOS(d)
        for atom in ped._lattice:
            if atom == ((0,), (1,)):
                assert ped[atom] == pytest.approx(1.0)
            elif atom == ((0, 1),):
                assert ped[atom] == pytest.approx(0.721928094887362)
            else:
                assert ped[atom] == pytest.approx(0.0)

    def test_and(self):
        d = bivariates["and"]
        ped = PED_MOS(d)
        for atom in ped._lattice:
            if atom == ((0,), (1,), (2,)):
                assert ped[atom] == pytest.approx(0.561278124459133)
            elif atom == ((0,), (1,)):
                assert ped[atom] == pytest.approx(0.438721875540867)
            elif atom in [((0, 1),), ((0, 1), (0, 2)), ((0, 1), (1, 2)), ((2,), (0, 1))]:
                assert ped[atom] == pytest.approx(0.25)
            else:
                assert ped[atom] == pytest.approx(0.0)

    def test_total_equals_joint_entropy(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        ped = PED_MOS(d)
        assert sum(ped._pis.values()) == pytest.approx(ped._total)

    def test_nonnegative(self):
        d = bivariates["and"]
        ped = PED_MOS(d)
        assert ped.nonnegative

    def test_consistent(self):
        d = bivariates["and"]
        ped = PED_MOS(d)
        assert ped.consistent

    def test_string_output(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        ped = PED_MOS(d)
        string = """\
+--------------------------+
|          H_mos           |
+--------+--------+--------+
| H_mos  |  H_r   |  H_d   |
+--------+--------+--------+
| {0:1}  | 1.7219 | 0.7219 |
|  {0}   | 1.0000 | 0.0000 |
|  {1}   | 1.0000 | 0.0000 |
| {0}{1} | 1.0000 | 1.0000 |
+--------+--------+--------+"""
        assert str(ped) == string

    def test_sources_kwarg(self):
        d = bivariates["and"]
        ped = PED_MOS(d, sources=((0,), (1,), (2,)))
        assert ped.nonnegative
        assert ped.consistent
        assert sum(ped._pis.values()) == pytest.approx(ped._total)


class TestPEDMOSPointwise:
    """Tests for the pointwise (per-outcome) computation."""

    def test_pointwise_nonnegative(self):
        d = bivariates["and"]
        ped = PED_MOS(d, pointwise=True)
        for node in ped._lattice:
            for o, v in ped.get_pw_pi(node).items():
                assert v >= -1e-10, f"Negative pw atom at {node}, {o}: {v}"

    def test_pointwise_expectation_matches_average(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        ped = PED_MOS(d, pointwise=True)
        for node in ped._lattice:
            pw_pi = ped.get_pw_pi(node)
            expected = sum(d[o] * pw_pi[o] for o in pw_pi)
            assert expected == pytest.approx(ped.get_pi(node))

    def test_pointwise_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        ped = PED_MOS(d, pointwise=True)
        intersection_node = ((0,), (1,))
        pw_pi = ped.get_pw_pi(intersection_node)
        for o in pw_pi:
            assert pw_pi[o] == pytest.approx(1.0)

    def test_pw_to_string(self):
        d = D(["00", "11"], [1 / 2] * 2)
        ped = PED_MOS(d, pointwise=True)
        s = ped.pw_to_string()
        assert "pointwise" in s
        assert "h_r" in s

    def test_pw_to_string_not_computed(self):
        d = D(["00", "11"], [1 / 2] * 2)
        ped = PED_MOS(d, pointwise=False)
        s = ped.pw_to_string()
        assert "not computed" in s

    def test_pointwise_trivariate(self):
        d = bivariates["and"]
        ped = PED_MOS(d, pointwise=True)
        for node in ped._lattice:
            pw_pi = ped.get_pw_pi(node)
            expected = sum(d[o] * pw_pi[o] for o in pw_pi)
            assert expected == pytest.approx(ped.get_pi(node))
