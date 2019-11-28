"""
Tests for dit.multivariate.secret_key_agreement.trivial_bounds.
"""
from __future__ import division

import pytest

import dit
from dit.multivariate import (upper_intrinsic_total_correlation,
                              upper_intrinsic_dual_total_correlation,
                              upper_intrinsic_caekl_mutual_information,
                              )


dist = dit.modify_outcomes(dit.example_dists.giant_bit(4, 2).__matmul__(dit.example_dists.n_mod_m(4, 2)),
                           lambda o: tuple(a+b for a, b in zip(o[:4], o[4:])))


def test_uitc1():
    """
    Test against known value.
    """
    value = upper_intrinsic_total_correlation(dist, dist.rvs[:-1], dist.rvs[-1])
    assert value == pytest.approx(1.0)


def test_uidtc1():
    """
    Test against known value.
    """
    value = upper_intrinsic_dual_total_correlation(dist, dist.rvs[:-1], dist.rvs[-1])
    assert value == pytest.approx(1.0)


def test_uicmi1():
    """
    Test against known value.
    """
    value = upper_intrinsic_caekl_mutual_information(dist, dist.rvs[:-1], dist.rvs[-1])
    assert value == pytest.approx(0.5)
