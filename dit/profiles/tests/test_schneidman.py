"""
Tests for dit.profiles.SchneidmanProfile. Known examples taken from http://arxiv.org/abs/1409.4708 .
"""

from __future__ import division

from nose.tools import assert_dict_equal
from numpy.testing import assert_array_almost_equal

from dit import Distribution
from dit.profiles import SchneidmanProfile

ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)
ex2 = Distribution(['000', '111'], [1/2]*2)
ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)
ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)
examples = [ex1, ex2, ex3, ex4]


def test_schneidman_profile():
    """
    Test against known examples.
    """
    profs = [(0.0, 0.0, 0.0),
             (0.0, 2.0, 0.0),
             (0.0, 1.0, 0.0),
             (0.0, 0.0, 1.0)]
    for ex, prof in zip(examples, profs):
        sp = SchneidmanProfile(ex)
        yield assert_array_almost_equal, [sp.profile[i] for i in (1,2,3)], prof
