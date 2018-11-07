"""
Tests for dit.multivariate.secret_key_agreement.two_part_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import intrinsic_1, intrinsic_2, intrinsic_3
from dit.multivariate.secret_key_agreement import (
    two_part_intrinsic_total_correlation,
    two_part_intrinsic_dual_total_correlation,
    two_part_intrinsic_CAEKL_mutual_information
)


@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_tpitc1(dist):
    """
    """
    tpitc = two_part_intrinsic_total_correlation(dist, [[0], [1]], [2])
    assert tpitc == pytest.approx(dist.secret_rate)


@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_tpidtc1(dist):
    """
    """
    tpidtc = two_part_intrinsic_dual_total_correlation(dist, [[0], [1]], [2])
    assert tpidtc == pytest.approx(dist.secret_rate)


@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_tpicaekl1(dist):
    """
    """
    tpicaekl = two_part_intrinsic_CAEKL_mutual_information(dist, [[0], [1]], [2])
    assert tpicaekl == pytest.approx(dist.secret_rate)
