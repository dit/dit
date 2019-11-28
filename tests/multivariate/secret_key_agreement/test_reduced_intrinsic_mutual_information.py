"""
Tests for dit.multivariate.secret_key_agreement.reduced_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import (reduced_intrinsic_total_correlation,
                              reduced_intrinsic_dual_total_correlation,
                              reduced_intrinsic_CAEKL_mutual_information,
                              )


@pytest.mark.skip(reason="Calculation of RIMI is prohibitively slow.")
@pytest.mark.parametrize(('dist', 'val'), [(intrinsic_1, 0.0), (intrinsic_2, 1.0), (intrinsic_3, 1.0)])
def test_1(dist, val):
    """
    Test against known values.
    """
    rimi = reduced_intrinsic_total_correlation(dist, [[0], [1]], [2], bound=(4,))
    assert rimi == pytest.approx(val, abs=1e-5)


@pytest.mark.skip(reason="Calculation of RIMI is prohibitively slow.")
@pytest.mark.parametrize(('dist', 'val'), [(intrinsic_1, 0.0), (intrinsic_2, 1.0), (intrinsic_3, 1.0)])
def test_2(dist, val):
    """
    Test against known values.
    """
    rimi = reduced_intrinsic_dual_total_correlation(dist, [[0], [1]], [2], bound=(4,))
    assert rimi == pytest.approx(val, abs=1e-5)


@pytest.mark.skip(reason="Calculation of RIMI is prohibitively slow.")
@pytest.mark.parametrize(('dist', 'val'), [(intrinsic_1, 0.0), (intrinsic_2, 1.0), (intrinsic_3, 1.0)])
def test_3(dist, val):
    """
    Test against known values.
    """
    rimi = reduced_intrinsic_CAEKL_mutual_information(dist, [[0], [1]], [2], bound=(4,))
    assert rimi == pytest.approx(val, abs=1e-5)
