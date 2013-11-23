"""
Tests for dit.example_dists.numeric.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_equal, assert_raises

import numpy as np

from dit.shannon import entropy
from dit.example_dists import bernoulli, binomial, hypergeometric, uniform

def test_bernoulli1():
    """ Test bernoulli distribution """
    d = bernoulli(1/2)
    assert_equal(d.outcomes, (0, 1))
    assert_almost_equal(sum(d.pmf), 1)

def test_bernoulli2():
    """ Test bernoulli distribution """
    for p in [ i/10 for i in range(0, 11)]:
        d = bernoulli(p)
        assert_almost_equal(d[0], 1-p)
        assert_almost_equal(d[1], p)

def test_bernoulli3():
    """ Test bernoulli distribution failures """
    for p in [-1, 1.5, 'a', int, []]:
        assert_raises(ValueError, bernoulli, p)

def test_binomial1():
    """ Test binomial distribution """
    for n in range(1, 10):
        d = binomial(n, 1/2)
        assert_equal(d.outcomes, tuple(range(n+1)))
        assert_almost_equal(sum(d.pmf), 1)

def test_binomial2():
    """ Test binomial distribution failures """
    for n in [-1, 1.5, 'a', int, []]:
        assert_raises(ValueError, binomial, n, 1/2)

def test_uniform1():
    """ Test uniform distribution """
    for n in range(2, 10):
        d = uniform(n)
        assert_equal(d.outcomes, tuple(range(n)))
        assert_almost_equal(d[0], 1/n)
        assert_almost_equal(entropy(d), np.log2(n))

def test_uniform2():
    """ Test uniform distribution failures """
    for v in [-1, 1.5, 'a', int, []]:
        assert_raises(ValueError, uniform, v)

def test_uniform3():
    """ Test uniform distribution construction """
    _as = [1, 2, 3, 4, 5]
    _bs = [5, 7, 9, 11, 13]
    for a, b in zip(_as, _bs):
        d = uniform(a, b)
        assert_equal(len(d.outcomes), b-a)
        assert_almost_equal(d[a], 1/(b-a))

def test_uniform4():
    """ Test uniform distribution failures """
    assert_raises(ValueError, uniform, 2, 0)

def test_uniform5():
    """ Test uniform distribution failures """
    assert_raises(ValueError, uniform, 0, [])

def test_hypergeometric1():
    """ Test hypergeometric distribution """
    d = hypergeometric(50, 5, 10)
    assert_almost_equal(d[4], 0.003964583)
    assert_almost_equal(d[5], 0.0001189375)

def test_hypergeometric2():
    """ Test hypergeometric distribution failures """
    vals = [(50, 5, -1), (50, -1, 10), (-1, 5, 10),
            (50, 5, 1.5), (50, 1.5, 10), (1.5, 5, 10),
            (50, 5, 'a'), (50, 'a', 10), ('a', 5, 10),
            (50, 5, int), (50, int, 10), (int, 5, 10),
            (50, 5, []), (50, [], 10), ([], 5, 10),
            ]
    for val in vals:
        assert_raises(ValueError, hypergeometric, *val)
