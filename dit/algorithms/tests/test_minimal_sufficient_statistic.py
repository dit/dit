"""
Tests for dit.algorithms.minimal_sufficient_statistic.
"""

from __future__ import division

from dit import Distribution, ScalarDistribution, pruned_samplespace
from dit.algorithms import insert_mss, mss, info_trim

def get_gm():
    """
    """
    outcomes = ['0101', '0110', '0111', '1010', '1011', '1101', '1110', '1111']
    pmf = [1/6, 1/12, 1/12, 1/6, 1/6, 1/6, 1/12, 1/12]
    return Distribution(outcomes, pmf)

def test_mss():
    """
    Test the construction of minimal sufficient statistics.
    """
    d = get_gm()
    d1 = mss(d, [0, 1], [2, 3])
    d2 = mss(d, [2, 3], [0, 1])
    dist = ScalarDistribution([0, 1], [1/3, 2/3])
    assert dist.is_approx_equal(d1)
    assert dist.is_approx_equal(d2)
    assert d1.is_approx_equal(d2)

def test_insert_mss():
    """
    Test the insertion of minimal sufficient statistics.
    """
    d = get_gm()
    d = insert_mss(d, -1, [0, 1], [2, 3])
    d = insert_mss(d, -1, [2, 3], [0, 1])
    d = d.marginal([4, 5])
    dist = pruned_samplespace(Distribution(['01', '10', '11'], [1/3, 1/3, 1/3]))
    assert d.is_approx_equal(dist)

def test_info_trim1():
    """

    """
    d1 = Distribution(['00', '01', '10', '11', '22', '33'], [1/8]*4+[1/4]*2)
    d2 = Distribution(['00', '11', '22'], [1/4, 1/4, 1/2])
    d3 = info_trim(d1)
    assert d3.is_approx_equal(d2)

def test_info_trim2():
    """

    """
    d1 = Distribution(['000', '001', '110', '111', '222', '333'], [1/8]*4+[1/4]*2)
    d2 = Distribution(['000', '111', '222', '332'], [1/4]*4)
    d3 = info_trim(d1)
    assert d3.is_approx_equal(d2)
