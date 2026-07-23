"""
Tests for dit.profiles.MFlatConnectedInformations.
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.divergences import kullback_leibler_divergence as D
from dit.profiles import (
    ConnectedDualInformations,
    ConnectedInformations,
    MFlatConnectedInformations,
)


def _weakly_dependent_binary():
    outcomes = ["".join(o) for o in product("01", repeat=3)]
    pmf = np.ones(8) / 8.0
    pmf[0] += 0.05
    pmf[7] += 0.05
    pmf = pmf / pmf.sum()
    return Distribution(outcomes, pmf)


def test_mflat_profile_pythagorean():
    d = _weakly_dependent_binary()
    prof = MFlatConnectedInformations(d, criterion="reverse_kl", eps=0.0)
    total = sum(prof.profile.values())
    assert total == pytest.approx(D(prof._dists[0], prof._dists[-1]), rel=1e-4, abs=1e-4)


def test_mflat_profile_keys():
    d = _weakly_dependent_binary()
    prof = MFlatConnectedInformations(d, criterion="jsd")
    assert set(prof.profile) == {1, 2, 3}


def test_mflat_jsd_nonnegative_atoms():
    d = _weakly_dependent_binary()
    prof = MFlatConnectedInformations(d, criterion="jsd", nrestarts=8)
    assert all(v >= -1e-6 for v in prof.profile.values())


def test_mflat_differs_from_schneidman_dual():
    """m-flat reverse-KL profile is not the MaxEnt DTC-increment profile."""
    d = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    mflat = MFlatConnectedInformations(d, criterion="reverse_kl")
    dual = ConnectedDualInformations(d)
    mvals = np.array([mflat.profile[i] for i in (1, 2, 3)])
    dvals = np.array([dual.profile[i] for i in (1, 2, 3)])
    assert not np.allclose(mvals, dvals, atol=1e-2)


def test_mflat_differs_from_connected_informations():
    d = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    mflat = MFlatConnectedInformations(d, criterion="jsd")
    conn = ConnectedInformations(d)
    mvals = np.array([mflat.profile[i] for i in (1, 2, 3)])
    cvals = np.array([conn.profile[i] for i in (1, 2, 3)])
    assert not np.allclose(mvals, cvals, atol=1e-2)


def test_giant_bit_and_xor_shapes_jsd():
    gb = Distribution(["000", "111"], [0.5, 0.5])
    xor = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    gb_p = MFlatConnectedInformations(gb, criterion="jsd", nrestarts=8)
    xor_p = MFlatConnectedInformations(xor, criterion="jsd", nrestarts=8)
    # Giant Bit: mass at pairwise order
    assert gb_p.profile[3] == pytest.approx(0.0, abs=1e-3)
    assert gb_p.profile[2] > 0.1
    # XOR: triplewise dominates
    assert xor_p.profile[3] > xor_p.profile[2]
    assert xor_p.profile[3] > xor_p.profile[1]


def test_w_has_nonzero_triplewise():
    w = Distribution(["001", "010", "100"], [1 / 3] * 3)
    prof = MFlatConnectedInformations(w, criterion="jsd", nrestarts=8)
    assert prof.profile[3] > 0.1


def test_copy_exact_at_order_2():
    copy = Distribution(["000", "001", "110", "111"], [0.25] * 4)
    prof = MFlatConnectedInformations(copy, criterion="jsd", nrestarts=8)
    assert prof.profile[3] == pytest.approx(0.0, abs=1e-3)
    assert prof.profile[2] > 0.1
