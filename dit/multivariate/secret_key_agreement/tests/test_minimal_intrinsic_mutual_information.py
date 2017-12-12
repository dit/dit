"""
Tests for dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import minimal_intrinsic_total_correlation

@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_1(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_total_correlation(dist, [[0], [1]], [2], bounds=(3,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-5)
