"""
Tests for dit.algorithms.mprojection (Amari m-flat m-projections).
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms import m_projection, mflat_mprojection_dists, symmetric_smooth
from dit.divergences import kullback_leibler_divergence as D
from dit.exceptions import ditException


def _weakly_dependent_binary():
    outcomes = ["".join(o) for o in product("01", repeat=3)]
    pmf = np.ones(8) / 8.0
    pmf[0] += 0.05
    pmf[7] += 0.05
    pmf = pmf / pmf.sum()
    return Distribution(outcomes, pmf)


def test_design_matrix_order_zero_is_constant():
    from dit.algorithms import mflat_design_matrix

    A, outcomes = mflat_design_matrix([("0", "1"), ("0", "1")], order=0)
    assert A.shape == (4, 1)
    assert np.allclose(A, 1.0)
    assert len(outcomes) == 4


def test_m_projection_order_zero_is_uniform():
    d = _weakly_dependent_binary()
    q = m_projection(d, 0, eps=0.0)
    assert np.allclose(q.pmf, 1.0 / len(q.outcomes))


def test_m_projection_order_n_recovers_target():
    d = _weakly_dependent_binary()
    q = m_projection(d, 3, eps=0.0)
    assert q.is_approx_equal(d, rtol=1e-10, atol=1e-10)


def test_jsd_projection_giant_bit():
    d = Distribution(["000", "111"], [0.5, 0.5])
    ladder = mflat_mprojection_dists(d, criterion="jsd")
    from dit.divergences.jensen_shannon_divergence import jensen_shannon_divergence

    # Exact by order 2
    assert jensen_shannon_divergence([d, ladder[2]]) == pytest.approx(0.0, abs=1e-5)
    assert jensen_shannon_divergence([d, ladder[1]]) > 0.1


def test_forward_kl_projection_copy():
    d = Distribution(["000", "001", "110", "111"], [0.25] * 4)
    q2 = m_projection(d, 2, criterion="forward_kl")
    assert D(d, q2) == pytest.approx(0.0, abs=1e-5)


def test_pythagorean_identity_full_support():
    d = _weakly_dependent_binary()
    ladder = mflat_mprojection_dists(d, eps=0.0)
    total = D(ladder[0], ladder[-1])
    parts = sum(D(ladder[i], ladder[i + 1]) for i in range(len(ladder) - 1))
    assert parts == pytest.approx(total, rel=1e-4, abs=1e-4)


def test_giant_bit_mass_in_pairwise():
    """Giant Bit is pairwise in the mixture sense: C*_3 ≈ 0, C*_2 dominates."""
    d = Distribution(["000", "111"], [0.5, 0.5])
    ladder = mflat_mprojection_dists(d, eps_schedule=(1e-4, 1e-6, 1e-8))
    c2 = D(ladder[1], ladder[2])
    c3 = D(ladder[2], ladder[3])
    assert c3 == pytest.approx(0.0, abs=1e-4)
    assert c2 > 1.0


def test_xor_mass_in_triplewise():
    """XOR is pure triplewise in the mixture sense."""
    d = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    ladder = mflat_mprojection_dists(d, eps_schedule=(1e-4, 1e-6, 1e-8))
    c2 = D(ladder[1], ladder[2])
    c3 = D(ladder[2], ladder[3])
    assert c2 < 0.1
    assert c3 > 1.0


def test_nonbinary_alphabet():
    """m-projection works for a ternary variable pair."""
    outcomes = [(a, b) for a in "012" for b in "012"]
    pmf = np.ones(9) / 9.0
    pmf[0] += 0.1
    pmf[8] += 0.1
    pmf /= pmf.sum()
    d = Distribution(outcomes, pmf)
    q1 = m_projection(d, 1, eps=0.0)
    assert D(q1, d) >= 0
    q2 = m_projection(d, 2, eps=0.0)
    assert D(q2, d) == pytest.approx(0.0, abs=1e-8)


def test_invalid_order_raises():
    d = _weakly_dependent_binary()
    with pytest.raises(ditException):
        m_projection(d, -1)


def test_symmetric_smooth_full_support():
    d = Distribution(["000", "111"], [0.5, 0.5])
    pe = symmetric_smooth(d, 1e-5)
    assert np.all(pe.pmf > 0)
    assert abs(pe.pmf.sum() - 1.0) < 1e-12


def test_symmetric_smooth_eps_zero_and_invalid():
    d = Distribution(["000", "111"], [0.5, 0.5])
    pe0 = symmetric_smooth(d, 0.0)
    assert abs(float(pe0["000"]) - 0.5) < 1e-12
    assert abs(float(pe0["111"]) - 0.5) < 1e-12
    with pytest.raises(ditException):
        symmetric_smooth(d, -0.1)
    with pytest.raises(ditException):
        symmetric_smooth(d, 1.0)


def test_unknown_criterion_raises():
    d = _weakly_dependent_binary()
    with pytest.raises(ditException):
        m_projection(d, 1, criterion="nope", eps=0.0)


def test_order_clamped_to_n():
    d = _weakly_dependent_binary()
    q = m_projection(d, 99, eps=0.0)
    assert q.is_approx_equal(d, rtol=1e-10, atol=1e-10)


def test_empty_subsets_is_uniform():
    from dit.algorithms import m_projection_from_subsets

    d = _weakly_dependent_binary()
    q = m_projection_from_subsets(d, subsets=[], criterion="jsd")
    assert np.allclose(q.pmf, 1.0 / len(q.outcomes))


def test_full_joint_subset_short_circuit():
    from dit.algorithms import m_projection_from_subsets

    d = Distribution(["000", "111"], [0.5, 0.5])
    q = m_projection_from_subsets(d, subsets=[(0, 1, 2)], criterion="jsd")
    # Full joint span recovers the (densified) target support masses.
    assert abs(float(q["000"]) + float(q["111"]) - 1.0) < 1e-8


def test_warm_start_and_eps_limit_path():
    from dit.algorithms import m_projection_eps_limit

    d = Distribution(["000", "111"], [0.5, 0.5])
    warm = m_projection(d, 1, eps=1e-4, nrestarts=4, maxiter=400)
    res = m_projection_eps_limit(
        d,
        order=2,
        eps_schedule=(1e-3, 1e-5),
        nrestarts=4,
        maxiter=400,
        warm_start=warm,
    )
    assert "path" in res and len(res["path"]) == 2
    assert res["rKL"] == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ditException):
        m_projection_eps_limit(d, order=1, eps_schedule=())


def test_mflat_design_matrix_order_guards():
    from dit.algorithms import mflat_design_matrix

    with pytest.raises(ditException):
        mflat_design_matrix([("0", "1"), ("0", "1")], order=-1)
    A, outcomes = mflat_design_matrix([("0", "1"), ("0", "1")], order=99)
    assert len(outcomes) == 4
    assert A.shape[0] == 4


def test_mflat_subsets_with_index_map():
    from dit.algorithms.mprojection import mflat_subsets_from_dependency

    node = frozenset([frozenset(["X", "Y"]), frozenset(["Z"])])
    subs = mflat_subsets_from_dependency(node, index_map={"X": 0, "Y": 1, "Z": 2})
    assert (0, 1) in subs
    assert (2,) in subs


def test_mflat_ladder_kmax_guards():
    d = _weakly_dependent_binary()
    with pytest.raises(ditException):
        mflat_mprojection_dists(d, k_max=-1)
    ladder = mflat_mprojection_dists(d, k_max=99, criterion="jsd", nrestarts=4)
    assert len(ladder) == 4  # orders 0..3
