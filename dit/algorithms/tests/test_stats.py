"""
Tests for dit.algorithms.stats
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises, assert_true
from numpy.testing import assert_array_almost_equal

from itertools import product

from math import ceil, floor

import numpy as np

from dit import Distribution as D
from dit.example_dists import binomial
from dit.algorithms import mean, median, mode, standard_deviation, \
                           standard_moment
from dit.algorithms.stats import _numerical_test

def test__numerical_test1():
    """ test _numerical_test on a good distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert(_numerical_test(d) is None)

def test__numerical_test2():
    """ Test _numerical_test on a bad distribution """
    d = D([(0, 0), (1, '0'), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert_raises(TypeError, _numerical_test, d)

def test_mean1():
    """ Test mean on binomial distribution """
    ns = range(2, 10)
    ps = np.linspace(0, 1, 11)
    for n, p in product(ns, ps):
        d = binomial(n, p)
        yield assert_almost_equal, mean(d), n*p

def test_mean2():
    """ Test mean on a generic distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert_array_almost_equal(mean(d), [2, 3/4])

def test_median1():
    """ Test median on binomial distribution """
    ns = range(2, 10)
    ps = np.linspace(0, 1, 11)
    for n, p in product(ns, ps):
        d = binomial(n, p)
        yield assert_true, median(d) in [floor(n*p), n*p, ceil(n*p)]

def test_median2():
    """ Test median on a generic distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    assert_array_almost_equal(median(d), [2, 1])

def test_mode1():
    """ Test mode on binomial distribution """
    ns = range(2, 10)
    ps = np.linspace(0, 1, 11)
    for n, p in product(ns, ps):
        d = binomial(n, p)
        yield assert_true, mode(d)[0][0] in [floor((n+1)*p), floor((n+1)*p)-1]

def test_mode2():
    """ Test mode on a generic distribution """
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1/8, 1/8, 3/8, 3/8])
    modes = [np.array([2, 3]), np.array([1])]
    for m1, m2 in zip(mode(d), modes):
        yield assert_array_almost_equal, m1, m2

def test_standard_deviation1():
    """ Test standard_deviation on binomial distribution """
    ns = range(2, 10)
    ps = np.linspace(0, 1, 11)
    for n, p in product(ns, ps):
        d = binomial(n, p)
        yield assert_almost_equal, standard_deviation(d), np.sqrt(n*p*(1-p))


def test_standard_moment1():
    """ Test standard_moment on binomial distribution """
    ns = range(3, 10)
    ps = np.linspace(0.1, 0.9, 9)
    for n, p in product(ns, ps):
        d = binomial(n, p)
        for i, m in {1: 0, 2: 1, 3: (1-2*p)/np.sqrt(n*p*(1-p))}.items():
            yield assert_almost_equal, standard_moment(d, i), m, 5
