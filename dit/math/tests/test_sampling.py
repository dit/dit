"""
Tests for dit.math.sampling.
"""

from nose.tools import assert_equal, assert_raises

import numpy as np

import dit.math.sampling as module
import dit.example_dists
from dit.exceptions import ditException

#sample(dist, size=None, rand=None, prng=None):
def test_sample1():
    # Basic sample
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = module.sample(d)
    assert_equal(x, '101')

    # with log dist
    dit.math.prng.seed(0)
    d.set_base(3.5)
    x = module.sample(d)
    assert_equal(x, '101')


def test_sample2():
    # Specified prng
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = module.sample(d, prng=dit.math.prng)
    assert_equal(x, '101')

def test_sample3():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = module.sample(d, rand=.3)
    assert_equal(x, '011')

def test_sample4():
    # More than one random number
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = module.sample(d, 6)
    assert_equal(x, ['101', '101', '101', '101', '011', '101'])

def test_sample5():
    # Bad prng
    d = dit.example_dists.Xor()
    assert_raises(ditException, module.sample, d, prng=3)

def test_sample6():
    # Not enough rands
    d = dit.example_dists.Xor()
    assert_raises(ditException, module.sample, d, 5, rand=[.1]*3)

def test_sample_discrete_python1():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = module._sample_discrete__python(d.pmf, .5)
    assert_equal(x, 2)

def test_sample_discrete_python2():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = module._samples_discrete__python(d.pmf, np.array([.5, .3, .2]))
    np.testing.assert_allclose(x, np.array([2, 1, 0]))

def test_ball_smoke():
    dit.math.prng.seed(0)
    x = module.ball(3)
    x_ = np.array([ 0.54094267,  0.45590783,  0.4121294 ])
    np.testing.assert_allclose(x, x_)

    dit.math.prng.seed(0)
    x = module.ball(3, prng=dit.math.prng)
    np.testing.assert_allclose(x, x_)

def test_base_with_size():
    # size 3, 1
    dit.math.prng.seed(0)
    x = module.ball(3, 1)
    x_ = np.array([[ 0.54094267,  0.45590783,  0.4121294 ]])
    np.testing.assert_allclose(x, x_)

    # size 4, 4
    dit.math.prng.seed(0)
    x = module.ball(4, 4)
    x_ = np.array([
        [ 0.27564989,  0.42024932,  0.28471498,  0.58022978],
        [ 0.61209883,  0.24355419,  0.50288752,  0.33594321],
        [ 0.4394934 ,  0.71612975,  0.05496026,  0.06741153],
        [ 0.0115166 ,  0.47426856,  0.44324584,  0.49556759]
    ])
    np.testing.assert_allclose(x, x_)

def test_2ball():
    dit.math.prng.seed(0)
    x = module._2ball(3, dit.math.prng)
    x_ = np.array([
        [-0.93662222, -0.29662586],
        [-0.28542405, -1.28408855],
        [-1.28553692, -0.96853918]])
    assert_equal(x.shape, (3, 2))
    np.testing.assert_allclose(x, x_)

def test_3ball_cylinder():
    dit.math.prng.seed(0)
    x = module._3ball_cylinder(3, dit.math.prng)
    x_ = np.array([
        [-0.7520198 ,  0.31101413,  0.08976637],
        [ 0.68515146, -0.55406718, -0.1526904 ],
        [ 0.77215823, -0.17942272,  0.29178823]
    ])
    assert_equal(x.shape, (3, 3))
    np.testing.assert_allclose(x, x_)
