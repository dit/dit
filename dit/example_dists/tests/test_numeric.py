"""
Tests for dit.example_dists.numeric.
"""

from __future__ import division

import pytest

import numpy as np

from dit.shannon import entropy
from dit.example_dists import bernoulli, binomial, hypergeometric, uniform

def test_bernoulli1():
    """ Test bernoulli distribution """
    d = bernoulli(1/2)
    assert d.outcomes == (0, 1)
    assert sum(d.pmf) == pytest.approx(1)

@pytest.mark.parametrize('p', [i/10 for i in range(0, 11)])
def test_bernoulli2(p):
    """ Test bernoulli distribution """
    d = bernoulli(p)
    assert d[0] == pytest.approx(1-p)
    assert d[1] == pytest.approx(p)

@pytest.mark.parametrize('p', [-1, 1.5, 'a', int, []])
def test_bernoulli3(p):
    """ Test bernoulli distribution failures """
    with pytest.raises(ValueError):
        bernoulli(p)

@pytest.mark.parametrize('n', range(1, 10))
def test_binomial1(n):
    """ Test binomial distribution """
    d = binomial(n, 1/2)
    assert d.outcomes == tuple(range(n+1))
    assert sum(d.pmf) == pytest.approx(1)

@pytest.mark.parametrize('n', [-1, 1.5, 'a', int, []])
def test_binomial2(n):
    """ Test binomial distribution failures """
    with pytest.raises(ValueError):
        binomial(n, 1/2)

def test_uniform1():
    """ Test uniform distribution """
    for n in range(2, 10):
        d = uniform(n)
        assert d.outcomes == tuple(range(n))
        assert d[0] == pytest.approx(1/n)
        assert entropy(d) == pytest.approx(np.log2(n))

@pytest.mark.parametrize('v', [-1, 1.5, 'a', int, []])
def test_uniform2(v):
    """ Test uniform distribution failures """
    with pytest.raises(ValueError):
        uniform(v)

@pytest.mark.parametrize(('a', 'b'), zip([1, 2, 3, 4, 5], [5, 7, 9, 11, 13]))
def test_uniform3(a, b):
    """ Test uniform distribution construction """
    d = uniform(a, b)
    assert len(d.outcomes) == b-a
    assert d[a] == pytest.approx(1/(b-a))

@pytest.mark.parametrize(('a', 'b'), [(2, 0), (0, [])])
def test_uniform4(a, b):
    """ Test uniform distribution failures """
    with pytest.raises(ValueError):
        uniform(a, b)

def test_hypergeometric1():
    """ Test hypergeometric distribution """
    d = hypergeometric(50, 5, 10)
    assert d[4] == pytest.approx(0.003964583)
    assert d[5] == pytest.approx(0.0001189375)

@pytest.mark.parametrize('vals', [
    (50, 5, -1),
    (50, -1, 10),
    (-1, 5, 10),
    (50, 5, 1.5),
    (50, 1.5, 10),
    (1.5, 5, 10),
    (50, 5, 'a'),
    (50, 'a', 10),
    ('a', 5, 10),
    (50, 5, int),
    (50, int, 10),
    (int, 5, 10),
    (50, 5, []),
    (50, [], 10),
    ([], 5, 10),
])
def test_hypergeometric2(vals):
    """ Test hypergeometric distribution failures """
    with pytest.raises(ValueError):
        hypergeometric(*vals)
