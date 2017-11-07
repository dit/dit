"""
Tests for dit.algorithms.stats
"""

from __future__ import division

import pytest

from hypothesis import given

from math import ceil, floor

import numpy as np

from dit import Distribution as D
from dit.algorithms import mean, median, mode, standard_deviation, \
                           standard_moment, maximum_correlation
from dit.algorithms.stats import _numerical_test
from dit.example_dists import binomial, dyadic, triadic
from dit.exceptions import ditException
from dit.utils.testing import distributions

def test__numerical_test1():
    """ test _numerical_test on a good distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert _numerical_test(d) is None

def test__numerical_test2():
    """ Test _numerical_test on a bad distribution """
    # A bad distribution is one with a non-numerical alphabet
    d = D([(0, '0'), (1, '0'), (2, '1'), (3, '1')], [1/8, 1/8, 3/8, 3/8])
    with pytest.raises(TypeError):
        _numerical_test(d)

@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_mean1(n, p):
    """ Test mean on binomial distribution """
    d = binomial(n, p)
    assert mean(d) == pytest.approx(n*p)

def test_mean2():
    """ Test mean on a generic distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert np.allclose(mean(d), [2, 3/4])

@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_median1(n, p):
    """ Test median on binomial distribution """
    d = binomial(n, p)
    assert median(d) in [floor(n*p), n*p, ceil(n*p)]

def test_median2():
    """ Test median on a generic distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert np.allclose(median(d), [2, 1])

@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_mode1(n, p):
    """ Test mode on binomial distribution """
    d = binomial(n, p)
    assert mode(d)[0][0] in [floor((n+1)*p), floor((n+1)*p)-1]

def test_mode2():
    """ Test mode on a generic distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    modes = [np.array([2, 3]), np.array([1])]
    for m1, m2 in zip(mode(d), modes):
        assert np.allclose(m1, m2)

@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_standard_deviation1(n, p):
    """ Test standard_deviation on binomial distribution """
    d = binomial(n, p)
    assert standard_deviation(d) == pytest.approx(np.sqrt(n*p*(1-p)))

@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize("p", np.linspace(0.1, 0.9, 9))
def test_standard_moment1(n, p):
    """ Test standard_moment on binomial distribution """
    d = binomial(n, p)
    for i, m in {1: 0, 2: 1, 3: (1-2*p)/np.sqrt(n*p*(1-p))}.items():
        assert standard_moment(d, i) == pytest.approx(m)

@pytest.mark.parametrize(('rvs', 'crvs'), [
    ([[0], [1]], []),
    ([[0], [1]], [2]),
])
@pytest.mark.parametrize('dist', [dyadic, triadic])
def test_maximum_correlation(dist, rvs, crvs):
    """ Test against known values """
    assert maximum_correlation(dist, rvs, crvs) == pytest.approx(1.0)

@pytest.mark.parametrize('rvs', [['X', 'Y', 'Z'], ['X']])
def test_maximum_correlation_failure(rvs):
    """ Test that maximum_correlation fails with len(rvs) != 2 """
    with pytest.raises(ditException):
        maximum_correlation(dyadic, rvs)

@given(dist1=distributions(alphabets=((2, 4),)*2, nondegenerate=True),
       dist2=distributions(alphabets=((2, 4),)*2, nondegenerate=True))
def test_maximum_correlation_tensorization(dist1, dist2):
    """
    Test tensorization:
        rho(X X' : Y Y') = max(rho(X:Y), rho(X', Y'))
    """
    mixed = dist1.__matmul__(dist2)
    rho_mixed = maximum_correlation(mixed, [[0, 2], [1, 3]])
    rho_a = maximum_correlation(dist1, [[0], [1]])
    rho_b = maximum_correlation(dist2, [[0], [1]])
    assert rho_mixed == pytest.approx(max(rho_a, rho_b))
