"""
Tests for dit.algorithms.stats
"""

from __future__ import division

import pytest

from math import ceil, floor

import numpy as np

from dit import Distribution as D
from dit.algorithms import mean, median, mode, standard_deviation, standard_moment
from dit.algorithms.stats import _numerical_test
from dit.example_dists import binomial


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
        assert standard_moment(d, i) == pytest.approx(m, abs=1e-5)
