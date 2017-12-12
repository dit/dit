"""
Tests for dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import (minimal_intrinsic_total_correlation,
                              minimal_intrinsic_dual_total_correlation,
                              minimal_intrinsic_CAEKL_mutual_information,
                              )

@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_1(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_total_correlation(dist, [[0], [1]], [2], bounds=(4,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-5)

@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_2(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_dual_total_correlation(dist, [[0], [1]], [2], bounds=(4,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-5)

@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_3(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_CAEKL_mutual_information(dist, [[0], [1]], [2], bounds=(4,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-5)
