"""
Tests for dit.multivariate.secret_key_agreement.reduced_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import reduced_intrinsic_total_correlation

@pytest.skip
@pytest.mark.parametrize(('dist', 'val'), [(intrinsic_1, 0.0), (intrinsic_2, 1.0), (intrinsic_3, 1.0)])
def test_1(dist, val):
    """
    Test against known values.
    """
    rimi = reduced_intrinsic_total_correlation(dist, [[0], [1]], [2])
    assert rimi == pytest.approx(val, abs=1e-5)
