"""
Tests for dit.multivariate.intrinsic_mutual_information
"""
from __future__ import division

from nose.tools import assert_almost_equal, raises

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import total_correlation
from dit.multivariate.intrinsic_mutual_information import (intrinsic_total_correlation,
                                                           intrinsic_dual_total_correlation,
                                                           intrinsic_caekl_mutual_information,
                                                           intrinsic_mutual_information)

dist1 = Distribution(['000', '011', '101', '110', '222', '333'], [1/8]*4+[1/4]*2)
dist2 = Distribution(['000', '011', '101', '110', '222', '333'], [1/8]*4+[1/4]*2)
dist2.set_rv_names('XYZ')
dist3 = Distribution(['00000', '00101', '11001', '11100', '22220', '33330'], [1/8]*4+[1/4]*2)
dist4 = Distribution(['00000', '00101', '11001', '11100', '22220', '33330'], [1/8]*4+[1/4]*2)
dist4.set_rv_names('VWXYZ')

def test_itc1():
    """
    Test against standard result.
    """
    itc = intrinsic_total_correlation(dist1, [[0], [1]], [2])
    assert_almost_equal(itc, 0)
    itc = intrinsic_total_correlation(dist1, [[0], [2]], [1])
    assert_almost_equal(itc, 0)
    itc = intrinsic_total_correlation(dist1, [[1], [2]], [0])
    assert_almost_equal(itc, 0)
    itc = intrinsic_total_correlation(dist3, [[0,1], [2]], [3, 4])
    assert_almost_equal(itc, 0)

def test_itc2():
    """
    Test against standard result, with rv names.
    """
    itc = intrinsic_total_correlation(dist2, ['X', 'Y'], 'Z')
    assert_almost_equal(itc, 0)
    itc = intrinsic_total_correlation(dist2, ['X', 'Z'], 'Y')
    assert_almost_equal(itc, 0)
    itc = intrinsic_total_correlation(dist2, ['Y', 'Z'], 'X')
    assert_almost_equal(itc, 0)
    itc = intrinsic_total_correlation(dist4, ['VW', 'X'], 'YZ')
    assert_almost_equal(itc, 0)

def test_itc3():
    """
    Test multivariate
    """
    itc = intrinsic_total_correlation(dist3, [[0], [1], [2,3]], [4])
    assert_almost_equal(itc, 3.3306155324443121)

def test_itc4():
    """
    Test multivariate, with rv names
    """
    itc = intrinsic_total_correlation(dist4, ['V', 'W', 'XY'], 'Z')
    assert_almost_equal(itc, 3.3306155324443121)

def test_idtc1():
    """
    Test against standard result.
    """
    idtc = intrinsic_dual_total_correlation(dist1, [[0], [1]], [2])
    assert_almost_equal(idtc, 0)
    idtc = intrinsic_dual_total_correlation(dist1, [[0], [2]], [1])
    assert_almost_equal(idtc, 0)
    idtc = intrinsic_dual_total_correlation(dist1, [[1], [2]], [0])
    assert_almost_equal(idtc, 0)
    idtc = intrinsic_dual_total_correlation(dist3, [[0,1], [2]], [3, 4])
    assert_almost_equal(idtc, 0)

def test_idtc2():
    """
    Test against standard result, with rv names.
    """
    idtc = intrinsic_dual_total_correlation(dist2, ['X', 'Y'], 'Z')
    assert_almost_equal(idtc, 0)
    idtc = intrinsic_dual_total_correlation(dist2, ['X', 'Z'], 'Y')
    assert_almost_equal(idtc, 0)
    idtc = intrinsic_dual_total_correlation(dist2, ['Y', 'Z'], 'X')
    assert_almost_equal(idtc, 0)
    idtc = intrinsic_dual_total_correlation(dist4, ['VW', 'X'], 'YZ')
    assert_almost_equal(idtc, 0)

def test_idtc3():
    """
    Test multivariate
    """
    idtc = intrinsic_dual_total_correlation(dist3, [[0], [1], [2,3]], [4])
    assert_almost_equal(idtc, 1.6887218755408671)

def test_idtc4():
    """
    Test multivariate, with rv names
    """
    idtc = intrinsic_dual_total_correlation(dist4, ['V', 'W', 'XY'], 'Z')
    assert_almost_equal(idtc, 1.6887218755408671)

def test_icmi1():
    """
    Test against standard result.
    """
    icmi = intrinsic_caekl_mutual_information(dist1, [[0], [1]], [2])
    assert_almost_equal(icmi, 0)
    icmi = intrinsic_caekl_mutual_information(dist1, [[0], [2]], [1])
    assert_almost_equal(icmi, 0)
    icmi = intrinsic_caekl_mutual_information(dist1, [[1], [2]], [0])
    assert_almost_equal(icmi, 0)
    icmi = intrinsic_caekl_mutual_information(dist3, [[0,1], [2]], [3, 4])
    assert_almost_equal(icmi, 0)

def test_icmi2():
    """
    Test against standard result, with rv names.
    """
    icmi = intrinsic_caekl_mutual_information(dist2, ['X', 'Y'], 'Z')
    assert_almost_equal(icmi, 0)
    icmi = intrinsic_caekl_mutual_information(dist2, ['X', 'Z'], 'Y')
    assert_almost_equal(icmi, 0)
    icmi = intrinsic_caekl_mutual_information(dist2, ['Y', 'Z'], 'X')
    assert_almost_equal(icmi, 0)
    icmi = intrinsic_caekl_mutual_information(dist4, ['VW', 'X'], 'YZ')
    assert_almost_equal(icmi, 0)

def test_icmi3():
    """
    Test multivariate
    """
    icmi = intrinsic_caekl_mutual_information(dist3, [[0], [1], [2]], [3,4])
    assert_almost_equal(icmi, 0)

def test_icmi4():
    """
    Test multivariate, with rv names
    """
    icmi = intrinsic_caekl_mutual_information(dist4, ['V', 'W', 'X'], 'YZ', nhops=128)
    assert_almost_equal(icmi, 0)

def test_constructor():
    """
    Test the generic constructor.
    """
    test = intrinsic_mutual_information(total_correlation).functional()
    itc = intrinsic_total_correlation(dist1, [[0], [1]], [2])
    itc2 = test(dist1, [[0], [1]], [2])
    assert_almost_equal(itc, itc2, places=4)

@raises(ditException)
def test_imi_fail():
    """
    Test that things fail when not provided with a conditional variable.
    """
    intrinsic_total_correlation(dist1, [[0], [1], [2]])
