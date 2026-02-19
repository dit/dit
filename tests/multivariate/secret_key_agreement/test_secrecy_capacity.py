"""
Tests for dit.multivariate.secret_key_agreement.secrecy_capacity
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import total_correlation
from dit.multivariate.secret_key_agreement.secrecy_capacity import SecrecyCapacity, secrecy_capacity
from tests._backends import backends


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('backend', backends)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_sc_1(dist, backend):
    """
    Test against known values.
    """
    sc = secrecy_capacity(dist, [0], [1], [2], bound_u=2, backend=backend)
    assert sc == pytest.approx(dist.secret_rate, abs=1e-5)


def test_sc_2():
    """
    Test the distribution.
    """
    sc = SecrecyCapacity(intrinsic_1, [0], [1], [2], bound_u=2)
    sc.optimize(x0=sc.construct_random_initial())
    d = sc.construct_distribution()
    assert total_correlation(d, [[3], [1]]) - total_correlation(d, [[3], [2]]) == pytest.approx(0)
