"""
Tests for dit.others.cumulative_residual_entropy.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises
from numpy.testing import assert_array_almost_equal

from six.moves import range # pylint: disable=redefined-builtin

from itertools import combinations, product

from dit import (ScalarDistribution as SD,
                 Distribution as D)
from dit.algorithms.stats import mean, standard_deviation
from dit.other import (cumulative_residual_entropy as CRE,
                       generalized_cumulative_residual_entropy as GCRE,
                       conditional_cumulative_residual_entropy as CCRE,
                       conditional_generalized_cumulative_residual_entropy as CGCRE)
from dit.example_dists import uniform, Xor

def miwin():
    d3 = [1, 2,       5, 6, 7,    9]
    d4 = [1,    3, 4, 5,       8, 9]
    d5 = [   2, 3, 4,    6, 7, 8   ]
    d = D(list(product(d3, d4, d5)), [1/6**3]*6**3)
    return d

def conditional_uniform1():
    events = [ (a, b) for a, b, in product(range(5), range(5)) if a <= b ]
    probs = [ 1/(5-a)/5 for a, _ in events ]
    d = D(events, probs)
    return d

def conditional_uniform2():
    events = [ (a-2, b-2) for a, b, in product(range(5), range(5)) if a <= b ]
    probs = [ 1/(3-a)/5 for a, _ in events ]
    d = D(events, probs)
    return d

def test_cre_1():
    """
    Test the CRE against known values for several uniform distributions.
    """
    dists = [ uniform(-n//2, n//2) for n in range(2, 23, 2) ]
    results = [0.5, 0.81127812, 1.15002242, 1.49799845, 1.85028649, 2.20496373,
               2.56111354, 2.91823997, 3.27604979, 3.6343579, 3.99304129]
    for d, r in zip(dists, results):
        yield assert_almost_equal, r, CRE(d)

def test_cre_2():
    """
    Test the CRE of a multivariate distribution (CRE is of each marginal).
    """
    d = miwin()
    assert_array_almost_equal(CRE(d), [3.34415526, 3.27909534, 2.56831826])

def test_cre_3():
    """
    Test that the CRE fails when the events are not numbers.
    """
    dist = Xor()
    assert_raises(TypeError, CRE, dist)

def test_gcre_1():
    """
    Test the GCRE against known values for the uniform distribution.
    """
    dists = [ uniform(-n//2, n//2) for n in range(2, 23, 2) ]
    results = [0.5, 1.31127812, 2.06831826, 2.80927657, 3.54316518, 4.27328199,
               5.00113503, 5.72751654, 6.45288453, 7.17752308, 7.90161817]
    for d, r in zip(dists, results):
        yield assert_almost_equal, r, GCRE(d)

def test_gcre_32():
    """
    Test the GCRE of a multivariate distribution.
    """
    d = miwin()
    assert_array_almost_equal(GCRE(d), [3.34415526, 3.27909534, 2.56831826])

def test_gcre_3():
    """
    Test that the GCRE fails on non-numeric events.
    """
    dist = Xor()
    assert_raises(TypeError, GCRE, dist)

def test_gcre_4():
    """
    Test that equal-length uniform distributions all have the same GCRE.
    """
    gcres = [ GCRE(uniform(i, i+5)) for i in range(-5, 1) ]
    for gcre in gcres:
        yield assert_almost_equal, gcre, gcres[0]

def test_ccre_1():
    """
    Test that independent RVs have CCRE = CRE.
    """
    d = miwin()
    for crvs in combinations([0, 1, 2], 2):
        rv = (set([0, 1, 2]) - set(crvs)).pop()
        ccre1 = CCRE(d, rv, crvs)
        ccre2 = CCRE(d, rv)
        yield assert_almost_equal, CRE(d)[rv], mean(ccre1)
        yield assert_almost_equal, CRE(d)[rv], mean(ccre2)
        yield assert_almost_equal, standard_deviation(ccre1), 0

def test_ccre_2():
    """
    Test a correlated distribution.
    """
    d = conditional_uniform1()
    ccre = CCRE(d, 1, [0])
    uniforms = [ CRE(uniform(i)) for i in range(1, 6) ]
    assert_array_almost_equal(ccre.outcomes, uniforms)

def test_ccre_3():
    """
    Test a correlated distribution.
    """
    d = conditional_uniform2()
    ccre = CCRE(d, 1, [0])
    print(ccre)
    uniforms = sorted([ CRE(uniform(i-2, 3)) for i in range(5) ])
    assert_array_almost_equal(ccre.outcomes, uniforms)

def test_cgcre_1():
    """
    """
    d = conditional_uniform2()
    cgcre = CGCRE(d, 1, [0])
    uniforms = [ GCRE(uniform(i)) for i in range(1, 6) ]
    assert_array_almost_equal(cgcre.outcomes, uniforms)
