"""
Tests for dit.multivariate.secret_key_agreement.skar_lower_bounds
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import total_correlation, necessary_intrinsic_mutual_information, secrecy_capacity
from dit.multivariate.secret_key_agreement.skar_lower_bounds import SecrecyCapacity


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_nimi_1(dist):
    """
    Test against known values.
    """
    nimi = necessary_intrinsic_mutual_information(dist, [[0], [1]], [2], bound_u=2, bound_v=4)
    assert nimi == pytest.approx(dist.secret_rate, abs=1e-5)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_sc_1(dist):
    """
    Test against known values.
    """
    sc = secrecy_capacity(dist, [[0], [1]], [2], bound_u=2)
    assert sc == pytest.approx(dist.secret_rate, abs=1e-5)


def test_sc_2():
    """
    Test the distribution.
    """
    sc = SecrecyCapacity(intrinsic_1, [0], [1], [2], bound_u=2)
    sc.optimize(x0=sc.construct_random_initial())
    d = sc.construct_distribution()
    assert total_correlation(d, [[3], [1]]) - total_correlation(d, [[3], [2]]) == pytest.approx(0)
