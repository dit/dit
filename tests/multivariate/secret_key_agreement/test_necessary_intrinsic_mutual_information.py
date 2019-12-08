# -*- coding: utf-8 -*-

"""
Tests for dit.multivariate.secret_key_agreement.skar_lower_bounds
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import necessary_intrinsic_mutual_information


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_nimi_1(dist):
    """
    Test against known values.
    """
    nimi = necessary_intrinsic_mutual_information(dist, [[0], [1]], [2], bound_u=2, bound_v=4)
    assert nimi == pytest.approx(dist.secret_rate, abs=1e-5)
