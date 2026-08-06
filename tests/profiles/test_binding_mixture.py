"""
Tests for BindingMixtureProfile and SharedRandomnessDecomposition (profiles).
"""

import pytest

from dit import Distribution
from dit.multivariate import dual_total_correlation as B
from dit.profiles import BindingMixtureProfile, SharedRandomnessDecomposition

ex_gb = Distribution(["000", "111"], [1 / 2] * 2)
ex_xor = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
ex_copy = Distribution(["000", "001", "110", "111"], [1 / 4] * 4)


def test_profile_repr_keys():
    prof = BindingMixtureProfile(ex_gb, k_max=3, n_init=6, seed=0)
    assert set(prof.profile.keys()) <= {1, 2, 3}
    assert len(prof.dists) == len(prof.profile)


def test_w_saturates_at_k3():
    w = Distribution(["001", "010", "100"], [1 / 3] * 3)
    prof = BindingMixtureProfile(w, k_max=5, n_init=10, seed=0)
    assert sum(prof.profile.values()) == pytest.approx(B(w), abs=1e-3)
    assert prof.forward_kl[-1] == pytest.approx(0.0, abs=1e-5)


def test_shared_randomness_string():
    s = str(SharedRandomnessDecomposition(ex_copy))
    assert "B" in s
    assert "Shared Randomness" in s or "dependency" in s.lower()
