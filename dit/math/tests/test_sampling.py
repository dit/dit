from nose.tools import *

import numpy as np

import dit.math.sampling as s
import dit.example_dists
from dit.exceptions import ditException

#sample(dist, size=None, rand=None, prng=None):
def test_sample1():
    # Basic sample
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = s.sample(d)
    assert_equal(x, '101')

    # with log dist
    dit.math.prng.seed(0)
    d.set_base(3.5)
    x = s.sample(d)
    assert_equal(x, '101')


def test_sample2():
    # Specified prng
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = s.sample(d, prng=dit.math.prng)
    assert_equal(x, '101')

def test_sample3():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = s.sample(d, rand=.3)
    assert_equal(x, '011')

def test_sample4():
    # More than one random number
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = s.sample(d, 6)
    assert_equal(x, ['101', '101', '101', '101', '011', '101'])

def test_sample5():
    # Bad prng
    d = dit.example_dists.Xor()
    assert_raises(ditException, s.sample, d, prng=3)

def test_sample6():
    # Not enough rands
    d = dit.example_dists.Xor()
    assert_raises(ditException, s.sample, d, 5, rand=[.1]*3)

def test_sample_discrete_python1():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = s._sample_discrete__python(d.pmf, .5)
    assert_equal(x, 2)

def test_sample_discrete_python2():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = s._samples_discrete__python(d.pmf, np.array([.5, .3, .2]))
    np.testing.assert_allclose(x, np.array([2, 1, 0]))
