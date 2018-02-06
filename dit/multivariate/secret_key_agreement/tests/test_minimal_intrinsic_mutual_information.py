"""
Tests for dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_information
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.multivariate import entropy, total_correlation
from dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_informations import (
    minimal_intrinsic_total_correlation,
    minimal_intrinsic_dual_total_correlation,
    minimal_intrinsic_CAEKL_mutual_information,
    minimal_intrinsic_mutual_information_constructor,
)
from dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_informations import MinimalIntrinsicTotalCorrelation


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_1(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_total_correlation(dist, [[0], [1]], [2], bounds=(4,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-4)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_2(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_dual_total_correlation(dist, [[0], [1]], [2], bounds=(4,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-4)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('dist', [intrinsic_1, intrinsic_2, intrinsic_3])
def test_3(dist):
    """
    Test against known values.
    """
    mimi = minimal_intrinsic_CAEKL_mutual_information(dist, [[0], [1]], [2], bounds=(4,))
    assert mimi == pytest.approx(dist.secret_rate, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_dist():
    """
    Test that the construct dist is accurate.
    """
    mimi = MinimalIntrinsicTotalCorrelation(intrinsic_1, [[0], [1]], [2], bound=3)
    mimi.optimize()
    d = mimi.construct_distribution()
    assert entropy(d, [3]) == pytest.approx(1.5)


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_constructor():
    """
    Test the generic constructor.
    """
    test = minimal_intrinsic_mutual_information_constructor(total_correlation).functional()
    itc = minimal_intrinsic_total_correlation(intrinsic_1, [[0], [1]], [2], bounds=(2,))
    itc2 = test(intrinsic_1, [[0], [1]], [2], bounds=(2,))
    assert itc == pytest.approx(itc2, abs=1e-4)
