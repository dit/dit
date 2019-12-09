# -*- coding: utf-8 -*-

"""
Tests for dit.multivariate.secret_key_agreement.two_part_intrinsic_mutual_information
"""

import pytest

from dit.example_dists import giant_bit, n_mod_m
from dit.exceptions import ditException
from dit.multivariate.secret_key_agreement import (
    two_part_intrinsic_total_correlation,
    two_part_intrinsic_dual_total_correlation,
    two_part_intrinsic_CAEKL_mutual_information
)
from dit.multivariate.secret_key_agreement.base_skar_optimizers import InnerTwoPartIntrinsicMutualInformation


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(['dist', 'value'], [
    (giant_bit(3, 2), 0.0),
    # (n_mod_m(3, 2), 0.0),
])
def test_tpitc1(dist, value):
    """
    """
    tpitc = two_part_intrinsic_total_correlation(dist, [[0], [1]], [2], bound_j=2, bound_u=2, bound_v=2)
    assert tpitc == pytest.approx(value, abs=1e-6)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(['dist', 'value'], [
    (giant_bit(3, 2), 0.0),
    # (n_mod_m(3, 2), 0.0),
])
def test_tpidtc1(dist, value):
    """
    """
    tpidtc = two_part_intrinsic_dual_total_correlation(dist, [[0], [1]], [2], bound_j=2, bound_u=2, bound_v=2)
    assert tpidtc == pytest.approx(value, abs=1e-6)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(['dist', 'value'], [
    (giant_bit(3, 2), 0.0),
    # (n_mod_m(3, 2), 0.0),
])
def test_tpicaekl1(dist, value):
    """
    """
    tpicaekl = two_part_intrinsic_CAEKL_mutual_information(dist, [[0], [1]], [2], bound_j=2, bound_u=2, bound_v=2)
    assert tpicaekl == pytest.approx(value, abs=1e-6)


def test_tpimi_fail1():
    """
    Ensure an exception is raised if no conditional variable is supplied.
    """
    with pytest.raises(ditException):
        two_part_intrinsic_total_correlation(n_mod_m(3, 2), [[0], [1]])


def test_tpimi_fail2():
    """
    Ensure an exception is raised if no conditional variable is supplied.
    """
    with pytest.raises(ditException):
        InnerTwoPartIntrinsicMutualInformation(n_mod_m(3, 2), [[0], [1]])
