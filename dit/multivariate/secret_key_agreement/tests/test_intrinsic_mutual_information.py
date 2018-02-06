"""
Tests for dit.multivariate.intrinsic_mutual_information
"""

from __future__ import division

import pytest

from hypothesis import given, settings

import numpy as np

from dit import Distribution
from dit.example_dists.intrinsic import intrinsic_1, intrinsic_2, intrinsic_3
from dit.exceptions import ditException
from dit.multivariate import total_correlation
from dit.multivariate.secret_key_agreement import intrinsic_mutual_informations as IMI


from dit.utils.testing import distributions

dist1 = Distribution([(0,0,0), (0,1,1), (1,0,1), (1,1,0), (2,2,2), (3,3,3)], [1/8]*4+[1/4]*2)
dist2 = Distribution(['000', '011', '101', '110', '222', '333'], [1/8]*4+[1/4]*2)
dist2.set_rv_names('XYZ')
dist3 = Distribution(['00000', '00101', '11001', '11100', '22220', '33330'], [1/8]*4+[1/4]*2)
dist4 = Distribution(['00000', '00101', '11001', '11100', '22220', '33330'], [1/8]*4+[1/4]*2)
dist4.set_rv_names('VWXYZ')
dist5 = Distribution(['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111'], [1/8]*8)
dist6 = Distribution(['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111'], [1/8]*8)
dist6.set_rv_names('WXYZ')


@pytest.mark.flaky(reruns=5)
def test_itc1():
    """
    Test against standard result.
    """
    itc = IMI.intrinsic_total_correlation(dist1, [[0], [1]], [2])
    assert itc == pytest.approx(0)
    itc = IMI.intrinsic_total_correlation(dist1, [[0], [2]], [1])
    assert itc == pytest.approx(0)
    itc = IMI.intrinsic_total_correlation(dist1, [[1], [2]], [0])
    assert itc == pytest.approx(0)
    itc = IMI.intrinsic_total_correlation(dist3, [[0,1], [2]], [3, 4])
    assert itc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_itc2():
    """
    Test against standard result, with rv names.
    """
    itc = IMI.intrinsic_total_correlation(dist2, ['X', 'Y'], 'Z')
    assert itc == pytest.approx(0)
    itc = IMI.intrinsic_total_correlation(dist2, ['X', 'Z'], 'Y')
    assert itc == pytest.approx(0)
    itc = IMI.intrinsic_total_correlation(dist2, ['Y', 'Z'], 'X')
    assert itc == pytest.approx(0)
    itc = IMI.intrinsic_total_correlation(dist4, ['VW', 'X'], 'YZ')
    assert itc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_itc3():
    """
    Test multivariate
    """
    itc = IMI.intrinsic_total_correlation(dist5, [[0], [1], [2]], [3])
    assert itc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_itc4():
    """
    Test multivariate, with rv names
    """
    itc = IMI.intrinsic_total_correlation(dist6, ['W', 'X', 'Y'], 'Z')
    assert itc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_itc5():
    """
    Test with initial condition.
    """
    itc = IMI.IntrinsicTotalCorrelation(dist1, [[0], [1]], [2])
    itc.optimize(x0=np.eye(4).ravel(), niter=5)
    d = itc.construct_distribution()
    print(d)
    val = total_correlation(d, [[0], [1]], [3])
    assert val == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_idtc1():
    """
    Test against standard result.
    """
    idtc = IMI.intrinsic_dual_total_correlation(dist1, [[0], [1]], [2])
    assert idtc == pytest.approx(0)
    idtc = IMI.intrinsic_dual_total_correlation(dist1, [[0], [2]], [1])
    assert idtc == pytest.approx(0)
    idtc = IMI.intrinsic_dual_total_correlation(dist1, [[1], [2]], [0])
    assert idtc == pytest.approx(0)
    idtc = IMI.intrinsic_dual_total_correlation(dist3, [[0,1], [2]], [3, 4])
    assert idtc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_idtc2():
    """
    Test against standard result, with rv names.
    """
    idtc = IMI.intrinsic_dual_total_correlation(dist2, ['X', 'Y'], 'Z')
    assert idtc == pytest.approx(0)
    idtc = IMI.intrinsic_dual_total_correlation(dist2, ['X', 'Z'], 'Y')
    assert idtc == pytest.approx(0)
    idtc = IMI.intrinsic_dual_total_correlation(dist2, ['Y', 'Z'], 'X')
    assert idtc == pytest.approx(0)
    idtc = IMI.intrinsic_dual_total_correlation(dist4, ['VW', 'X'], 'YZ')
    assert idtc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_idtc3():
    """
    Test multivariate
    """
    idtc = IMI.intrinsic_dual_total_correlation(dist5, [[0], [1], [2]], [3])
    assert idtc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_idtc4():
    """
    Test multivariate, with rv names
    """
    idtc = IMI.intrinsic_dual_total_correlation(dist6, ['W', 'X', 'Y'], 'Z')
    assert idtc == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_icmi1():
    """
    Test against standard result.
    """
    icmi = IMI.intrinsic_caekl_mutual_information(dist1, [[0], [1]], [2])
    assert icmi == pytest.approx(0)
    icmi = IMI.intrinsic_caekl_mutual_information(dist1, [[0], [2]], [1])
    assert icmi == pytest.approx(0)
    icmi = IMI.intrinsic_caekl_mutual_information(dist1, [[1], [2]], [0])
    assert icmi == pytest.approx(0)
    icmi = IMI.intrinsic_caekl_mutual_information(dist3, [[0,1], [2]], [3, 4])
    assert icmi == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_icmi2():
    """
    Test against standard result, with rv names.
    """
    icmi = IMI.intrinsic_caekl_mutual_information(dist2, ['X', 'Y'], 'Z')
    assert icmi == pytest.approx(0)
    icmi = IMI.intrinsic_caekl_mutual_information(dist2, ['X', 'Z'], 'Y')
    assert icmi == pytest.approx(0)
    icmi = IMI.intrinsic_caekl_mutual_information(dist2, ['Y', 'Z'], 'X')
    assert icmi == pytest.approx(0)
    icmi = IMI.intrinsic_caekl_mutual_information(dist4, ['VW', 'X'], 'YZ')
    assert icmi == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_icmi3():
    """
    Test multivariate
    """
    icmi = IMI.intrinsic_caekl_mutual_information(dist5, [[0], [1], [2]], [3])
    assert icmi == pytest.approx(0)


@pytest.mark.flaky(reruns=5)
def test_icmi4():
    """
    Test multivariate, with rv names
    """
    icmi = IMI.intrinsic_caekl_mutual_information(dist6, ['W', 'X', 'Y'], 'Z')
    assert icmi == pytest.approx(0)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_constructor():
    """
    Test the generic constructor.
    """
    test = IMI.intrinsic_mutual_information_constructor(total_correlation).functional()
    itc = IMI.intrinsic_total_correlation(dist1, [[0], [1]], [2], bound=3)
    itc2 = test(dist1, [[0], [1]], [2], bound=3)
    assert itc == pytest.approx(itc2, abs=1e-4)


def test_imi_fail():
    """
    Test that things fail when not provided with a conditional variable.
    """
    with pytest.raises(ditException):
        IMI.intrinsic_total_correlation(dist1, [[0], [1], [2]])


@pytest.mark.flaky(rerun=5)
@settings(deadline=None)
@given(dist=distributions(alphabets=((2, 4),)*3))
def test_bounds(dist):
    """
    I[X:Y v Z] <= I[X:Y]
    I[X:Y v Z] <= I[X:Y|Z]
    """
    imi = IMI.intrinsic_total_correlation(dist, [[0], [1]], [2])
    mi = total_correlation(dist, [[0], [1]])
    cmi = total_correlation(dist, [[0], [1]], [2])
    assert imi <= mi + 1e-10
    assert imi <= cmi + 1e-10


@pytest.mark.parametrize(('dist', 'val'), [(intrinsic_1, 0.0), (intrinsic_2, 1.5), (intrinsic_3, 1.3932929108738521)])
def test_1(dist, val):
    """
    Test against known values.
    """
    imi = IMI.intrinsic_total_correlation(dist, [[0], [1]], [2])
    assert imi == pytest.approx(val, abs=1e-5)
