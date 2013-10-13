from __future__ import division

from nose.tools import *

import numpy as np

from dit.algorithms import entropy
from dit.example_dists import bernoulli, binomial, uniform

def test_bernoulli1():
    d = bernoulli(1/2)
    assert_equal(d.outcomes, (0,1))
    assert_almost_equal(sum(d.pmf), 1)

def test_bernoulli2():
    for p in [ i/10 for i in range(0, 11)]:
        d = bernoulli(p)
        assert_almost_equal(d[0], 1-p)
        assert_almost_equal(d[1], p)

def test_bernoulli3():
    for p in [-1, 1.5, 'a', int, []]:
        assert_raises(ValueError, bernoulli, p)

def test_binomial1():
    for n in range(1, 10):
        d = binomial(n, 1/2)
        assert_equal(d.outcomes, tuple(range(n+1)))
        assert_almost_equal(sum(d.pmf), 1)

def test_binomial2():
    for n in [-1, 1.5, 'a', int, []]:
        assert_raises(ValueError, binomial, n, 1/2)

def test_uniform1():
    for n in range(2, 10):
        d = uniform(n)
        assert_equal(d.outcomes, tuple(range(n)))
        assert_almost_equal(d[0], 1/n)
        assert_almost_equal(entropy(d), np.log2(n))