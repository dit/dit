"""
Tests for dit.algorithms.minimal_sufficient_statistic.
"""

from __future__ import division

from dit import Distribution, ScalarDistribution, pruned_samplespace
from dit.algorithms import insert_mss, mss, mss_sigalg

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
