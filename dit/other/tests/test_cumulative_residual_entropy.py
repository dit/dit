"""
Tests for dit.others.cumulative_residual_entropy.
"""

from __future__ import division

from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from six.moves import range # pylint: disable=redefined-builtin

from itertools import product

from dit import (ScalarDistribution as SD,
                 Distribution as D)
from dit.other import (cumulative_residual_entropy as CRE,
                       generalized_cumulative_residual_entropy as GCRE)
from dit.example_dists import uniform

def test_cre_1():
    """
    """
    dists = [ uniform(-n//2, n//2) for n in range(2, 23, 2) ]
    results = [0.5, 0.81127812, 1.15002242, 1.49799845, 1.85028649, 2.20496373,
               2.56111354, 2.91823997, 3.27604979, 3.6343579, 3.99304129]
    for d, r in zip(dists, results):
        yield assert_almost_equal, r, CRE(d)

def test_cre_2():
    """
    """
    d3 = [1, 2,       5, 6, 7,    9]
    d4 = [1,    3, 4, 5,       8, 9]
    d5 = [   2, 3, 4,    6, 7, 8   ]
    d = D(list(product(d3, d4, d5)), [1/6**3]*6**3)
    assert_array_almost_equal(CRE(d), [3.34415526, 3.27909534, 2.56831826])

def test_gcre_1():
    """
    """
    dists = [ uniform(-n//2, n//2) for n in range(2, 23, 2) ]
    results = [0.5, 1.31127812, 2.06831826, 2.80927657, 3.54316518, 4.27328199,
               5.00113503, 5.72751654, 6.45288453, 7.17752308, 7.90161817]
    for d, r in zip(dists, results):
        yield assert_almost_equal, r, CRE(d, generalized=True)

def test_gcre_2():
    """
    """
    dists = [ uniform(-n//2, n//2) for n in range(2, 23, 2) ]
    results = [0.5, 1.31127812, 2.06831826, 2.80927657, 3.54316518, 4.27328199,
               5.00113503, 5.72751654, 6.45288453, 7.17752308, 7.90161817]
    for d, r in zip(dists, results):
        yield assert_almost_equal, r, GCRE(d)
