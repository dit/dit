"""
Tests for dit.others.cumulative_residual_entropy.
"""

from __future__ import division

import pytest

from six.moves import range # pylint: disable=redefined-builtin

from itertools import combinations, product

import numpy as np

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

@pytest.mark.parametrize(('n', 'val'), zip(range(2, 23, 2),[0.5, 0.81127812, 1.15002242, 1.49799845,
                                                            1.85028649, 2.20496373, 2.56111354,
                                                            2.91823997, 3.27604979, 3.6343579,
                                                            3.99304129]))
def test_cre_1(n, val):
    """
    Test the CRE against known values for several uniform distributions.
    """
    dist = uniform(-n//2, n//2)
    assert CRE(dist) == pytest.approx(val)

def test_cre_2():
    """
    Test the CRE of a multivariate distribution (CRE is of each marginal).
    """
    d = miwin()
    assert np.allclose(CRE(d), [3.34415526, 3.27909534, 2.56831826])

def test_cre_3():
    """
    Test that the CRE fails when the events are not numbers.
    """
    dist = Xor()
    with pytest.raises(TypeError):
        CRE(dist)

@pytest.mark.parametrize(('n', 'val'), zip(range(2, 23, 2),[0.5, 1.31127812, 2.06831826, 2.80927657,
                                                            3.54316518, 4.27328199, 5.00113503,
                                                            5.72751654, 6.45288453, 7.17752308,
                                                            7.90161817]))
def test_gcre_1(n, val):
    """
    Test the GCRE against known values for the uniform distribution.
    """
    dist = uniform(-n//2, n//2)
    assert GCRE(dist) == pytest.approx(val)

def test_gcre_32():
    """
    Test the GCRE of a multivariate distribution.
    """
    d = miwin()
    assert np.allclose(GCRE(d), [3.34415526, 3.27909534, 2.56831826])

def test_gcre_3():
    """
    Test that the GCRE fails on non-numeric events.
    """
    dist = Xor()
    with pytest.raises(TypeError):
        GCRE(dist)

@pytest.mark.parametrize('i', range(-5, 1))
def test_gcre_4(i):
    """
    Test that equal-length uniform distributions all have the same GCRE.
    """
    gcre = GCRE(uniform(i, i+5))
    assert gcre == pytest.approx(GCRE(uniform(-5, 0)))

@pytest.mark.parametrize('crvs', combinations([0, 1, 2], 2))
def test_ccre_1(crvs):
    """
    Test that independent RVs have CCRE = CRE.
    """
    d = miwin()
    rv = (set([0, 1, 2]) - set(crvs)).pop()
    ccre1 = CCRE(d, rv, crvs)
    ccre2 = CCRE(d, rv)
    assert CRE(d)[rv] == pytest.approx(mean(ccre1))
    assert CRE(d)[rv] == pytest.approx(mean(ccre2))
    assert standard_deviation(ccre1) == pytest.approx(0)

def test_ccre_2():
    """
    Test a correlated distribution.
    """
    d = conditional_uniform1()
    ccre = CCRE(d, 1, [0])
    uniforms = [ CRE(uniform(i)) for i in range(1, 6) ]
    assert np.allclose(ccre.outcomes, uniforms)

def test_ccre_3():
    """
    Test a correlated distribution.
    """
    d = conditional_uniform2()
    ccre = CCRE(d, 1, [0])
    uniforms = sorted([ CRE(uniform(i-2, 3)) for i in range(5) ])
    assert np.allclose(ccre.outcomes, uniforms)

def test_cgcre_1():
    """
    Test the CGCRE against known values.
    """
    d = conditional_uniform2()
    cgcre = CGCRE(d, 1, [0])
    uniforms = [ GCRE(uniform(i)) for i in range(1, 6) ]
    assert np.allclose(cgcre.outcomes, uniforms)

@pytest.mark.parametrize('crvs', combinations([0, 1, 2], 2))
def test_cgcre_2(crvs):
    """
    Test that independent RVs have CGCRE = GCRE.
    """
    d = miwin()
    rv = (set([0, 1, 2]) - set(crvs)).pop()
    ccre1 = CGCRE(d, rv, crvs)
    ccre2 = CGCRE(d, rv)
    assert GCRE(d)[rv] == pytest.approx(mean(ccre1))
    assert GCRE(d)[rv] == pytest.approx(mean(ccre2))
    assert standard_deviation(ccre1) == pytest.approx(0)
