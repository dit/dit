"""
Tests for dit.profiles.MUIProfile. Known examples taken from http://arxiv.org/abs/1409.4708 .
"""

from __future__ import division

from nose.tools import assert_dict_equal
from numpy.testing import assert_array_almost_equal

from dit import Distribution
from dit.profiles import MUIProfile

ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)
ex2 = Distribution(['000', '111'], [1/2]*2)
ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)
ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)
examples = [ex1, ex2, ex3, ex4]


def test_mui_profile():
    """
    Test against known examples.
    """
    profs = [({0.0: 1.0}, [3.0]),
             ({0.0: 3.0}, [1.0]),
             ({0.0: 2.0, 1.0: 1.0}, [1.0, 1.0]),
             ({0.0: 3/2}, [2.0])]
    for ex, (prof, width) in zip(examples, profs):
        mui = MUIProfile(ex)
        yield assert_dict_equal, mui.profile, prof
        yield assert_array_almost_equal, mui.widths, width
