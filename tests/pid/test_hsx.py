"""
Tests for dit.pid.hsx.
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.example_dists import Xor
from dit.multivariate import entropy
from dit.pid.hsx import PED_SX, h_sx, h_sx_pw
from dit.utils.testing import distributions


class TestHSx:
    """Tests for the h_sx redundancy functional."""

    def test_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        assert h_sx(d, ((0,), (1,))) == pytest.approx(1.0)

    def test_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        assert h_sx(d, ((0,), (1,))) == pytest.approx(np.log2(4 / 3))

    def test_single_source_is_entropy(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        assert h_sx(d, ((0, 1),)) == pytest.approx(entropy(d))
        assert h_sx(d, ((0,),)) == pytest.approx(1.0)

    def test_pointwise(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        pw = h_sx_pw(d, ((0,), (1,)))
        assert pw[("0", "0")] == pytest.approx(-np.log2(0.6))
        assert pw[("0", "1")] == pytest.approx(-np.log2(0.9))


class TestPEDSX:
    """Tests for the PED_SX decomposition."""

    def test_independent_bits(self):
        ped = PED_SX(D(["00", "01", "10", "11"], [1 / 4] * 4))
        assert ped[((0,), (1,))] == pytest.approx(np.log2(4 / 3))
        assert ped[((0,),)] == pytest.approx(1 - np.log2(4 / 3))
        assert ped[((0, 1),)] == pytest.approx(np.log2(4 / 3))

    def test_xor_matches_syntropy(self):
        """
        Reference atoms from Syntropy's ``partial_entropy_decomposition(..., "hsx")``.
        """
        ped = PED_SX(Xor())
        assert ped[((0,), (1,))] == pytest.approx(0.4150374992788438)
        assert ped[((0,), (2,))] == pytest.approx(0.4150374992788438)
        assert ped[((0,), (1, 2))] == pytest.approx(0.16992500144231237)
        assert ped[((0, 1), (0, 2), (1, 2))] == pytest.approx(0.24511249783653133)
        assert ped[((0,), (1,), (2,))] == pytest.approx(0.0)

    def test_pointwise_averages_to_atoms(self):
        ped = PED_SX(Xor(), pointwise=True)
        d = ped._dist
        for node in ped._lattice:
            avg = sum(d[o] * v for o, v in ped.get_pw_pi(node).items())
            assert avg == pytest.approx(ped[node])
        assert "H_sx (pointwise)" in ped.pw_to_string()


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_atoms_sum_to_entropy(dist):
    ped = PED_SX(dist)
    assert sum(ped[node] for node in ped._lattice) == pytest.approx(entropy(dist), abs=1e-6)
