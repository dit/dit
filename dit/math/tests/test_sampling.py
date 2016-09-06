"""
Tests for dit.math.sampling.
"""

import pytest

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
    assert x == '101'

    # with log dist
    dit.math.prng.seed(0)
    d.set_base(3.5)
    x = module.sample(d)
    assert x == '101'

def test_sample2():
    # Specified prng
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = module.sample(d, prng=dit.math.prng)
    assert x == '101'

def test_sample3():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = module.sample(d, rand=.3)
    assert x == '011'

def test_sample4():
    # More than one random number
    d = dit.example_dists.Xor()
    dit.math.prng.seed(0)
    x = module.sample(d, 6)
    assert x == ['101', '101', '101', '101', '011', '101']

def test_sample5():
    # Bad prng
    d = dit.example_dists.Xor()
    with pytest.raises(ditException):
        module.sample(d, prng=3)

def test_sample6():
    # Not enough rands
    d = dit.example_dists.Xor()
    with pytest.raises(ditException):
        module.sample(d, 5, rand=[.1]*3)

def test_sample_discrete_python1():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = module._sample_discrete__python(d.pmf, .5)
    assert x == 2

def test_sample_discrete_python2():
    # Specified rand number
    d = dit.example_dists.Xor()
    x = module._samples_discrete__python(d.pmf, np.array([.5, .3, .2]))
    assert np.allclose(x, np.array([2, 1, 0]))

def test_ball_smoke():
    dit.math.prng.seed(0)
    x = module.ball(3)
    x_ = np.array([ 0.21324626,  0.4465436 , -0.65226253])
    assert np.allclose(x, x_)

    dit.math.prng.seed(0)
    x = module.ball(3, prng=dit.math.prng)
    assert np.allclose(x, x_)

def test_base_with_size():
    # size 3, 1
    dit.math.prng.seed(0)
    x = module.ball(3, 1)
    x_ = np.array([[ 0.21324626,  0.4465436 , -0.65226253]])
    assert np.allclose(x, x_)

    # size 4, 4
    dit.math.prng.seed(0)
    x = module.ball(4, 4)
    x_ = np.array([
        [ 0.69375635, -0.36303705,  0.35293677, -0.05622584],
        [-0.06238751,  0.24817385,  0.08706278,  0.87899164],
        [ 0.70592518,  0.1128636 ,  0.41171971,  0.30951042],
        [ 0.72885111, -0.1000816 ,  0.15272267, -0.41665039]
    ])
    assert np.allclose(x, x_)

def test_2ball():
    dit.math.prng.seed(0)
    x = module._2ball(3, dit.math.prng)
    x_ = np.array([
        [-0.93662222, -0.29662586],
        [-0.14853979, -0.66826269],
        [-0.31184301, -0.23494632]
    ])
    assert x.shape == (3, 2)
    assert np.allclose(x, x_)

def test_3ball_cylinder():
    dit.math.prng.seed(0)
    x = module._3ball_cylinder(3, dit.math.prng)
    x_ = np.array([
        [-0.7520198 ,  0.31101413,  0.08976637],
        [ 0.68515146, -0.55406718, -0.1526904 ],
        [ 0.77215823, -0.17942272,  0.29178823]
    ])
    assert x.shape == (3, 3)
    assert np.allclose(x, x_)

def test_norm_smoketest():
    d = np.array([.2, .3, .5])

    # prng is None
    dit.math.prng.seed(0)
    x = module.norm(d)
    x_ = np.array([ 0.49606291,  0.13201838,  0.37191871])
    assert np.allclose(x, x_)

    # prng is not None
    dit.math.prng.seed(0)
    x = module.norm(d, prng=dit.math.prng)
    assert np.allclose(x, x_)

    # size is not None
    dit.math.prng.seed(0)
    x = module.norm(d, size=1)
    assert np.allclose(x, np.asarray([x_]))

def test_norm_spherical_cov():
    d = np.array([.2, .3, .5])
    dit.math.prng.seed(0)
    x = module.norm(d, .3)
    x_ = np.array([ 0.34790127,  0.20240029,  0.44969844])
    assert np.allclose(x, x_)

def test_norm_diagonal_cov():
    d = np.array([.2, .3, .5])
    dit.math.prng.seed(0)
    x = dit.math.norm(d, np.array([.3, .5]))
    x_ = np.array([ 0.33458841,  0.40485058,  0.26056101])
    assert np.allclose(x, x_)

def test_norm_cov():
    d = np.array([.2, .3, .5])
    dit.math.prng.seed(0)
    ilrcov = np.array([[.3, .2],[.2, .4]])
    x = dit.math.norm(d, ilrcov, size=3)
    x_ = np.array([
        [ 0.07400846,  0.27603643,  0.64995511],
        [ 0.09984353,  0.44856934,  0.45158713],
        [ 0.07260608,  0.18948779,  0.73790613]
    ])
    assert np.allclose(x, x_)

def test_norm_badshape_cov():
    d = np.array([.2, .3, .5])
    dit.math.prng.seed(0)
    ilrcov = np.array([[.3, .5, .1],[.5, .4, .3], [.2, .5, .3]])
    with pytest.raises(ditException):
        dit.math.norm(d, ilrcov)

    ilrcov = np.array([.3, .5, .1])
    with pytest.raises(ditException):
        dit.math.norm(d, ilrcov)

    ilrcov = np.array(np.random.rand(16).reshape(2,2,4))
    with pytest.raises(ditException):
        dit.math.norm(d, ilrcov)

def test_norm_toomany():
    d = np.array([[.2, .3, .5], [.5, .2, .3]])
    with pytest.raises(ditException):
        dit.math.norm(d)

def test__annulus2_nosize():
    # When size is None, it should just return the sample.
    prng = np.random.RandomState()
    samples = dit.math.sampling._annulus2(0, 1, size=None, prng=prng)
    assert samples.shape == (2,)

def test__annulus2_size():
    # When size is not None, it should return an array of the samples.
    prng = np.random.RandomState()
    samples = dit.math.sampling._annulus2(0, 1, size=1, prng=prng)
    assert samples.shape == (1,2)
