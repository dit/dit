"""
Tests for dit.multivariate.secret_key_agreement.lower_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import lower_intrinsic_mutual_information


@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_1(dist):
    """
    Test against known values.
    """
    limi = lower_intrinsic_mutual_information(dist, [[0], [1]], [2])
    assert limi == pytest.approx(dist.secret_rate)
