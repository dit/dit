# coding: utf-8

from __future__ import division
from __future__ import print_function

import dit
import numpy as np

from nose.tools import *

module = dit.math.pmfops

def test_perturb():
    # Smoke test
    d = np.array([0, .5, .5])
    d2 = module.perturb(d, .00001)
    d3 = d2.round(2)
    np.testing.assert_allclose(d, d3)

def test_convex_combination():
    d1 = np.array([0, .5, .5])
    d2 = np.array([.5, .5, 0])
    d3_= np.array([.25, .5, .25])
    d3 = module.convex_combination(np.array([d1, d2]))
    np.testing.assert_allclose(d3, d3_)

def test_convex_combination_weights():
    d1 = np.array([0, .5, .5])
    d2 = np.array([.5, .5, 0])
    weights = [1, 0]
    d3 = module.convex_combination(np.array([d1, d2]), weights)
    np.testing.assert_allclose(d3, d1)

def test_downsample_onepmf():
    # One pmf
    d1 = np.array([0, .51, .49])
    d2_ = np.array([0, .5, .5])
    d2 = module.downsample(d1, 1)
    np.testing.assert_allclose(d2, d2_)

def test_downsample_twopmf():
    # Two pmf
    d1 = np.array([[0, .51, .49], [.6, .3, .1]])
    d2_ = np.array([[0, .5, .5], [.5, .5, 0]])
    d2 = module.downsample(d1, 1)
    np.testing.assert_allclose(d2, d2_)

def test_downsample_badmethod():
    d1 = np.array([0, .51, .49])
    assert_raises(
        NotImplementedError, module.downsample, d1, 3, method='whatever'
    )

def test_projections1():
    d = np.array([ 0.03231933,  0.89992681,  0.06775385])
    d2_ = np.array([
        [ 0.03231933,  0.89992681,  0.06775385],
        [ 0.        ,  0.92998325,  0.07001675],
        [ 0.        ,  0.875     ,  0.125     ]
    ])
    d2 = module.projections(d, 3)
    np.testing.assert_allclose(d2, d2_)

def test_projections2():
    d = np.array([ 0.51,  0.48,  0.01])
    d2_ = np.array([
        [ 0.51      ,  0.48      ,  0.01      ],
        [ 0.5       ,  0.48979592,  0.01020408],
        [ 0.5       ,  0.5       ,  0.        ]
    ])
    d2 = module.projections(d, 3)
    np.testing.assert_allclose(d2, d2_, rtol=1e-7, atol=1e-8)

def test_clamps():
    d = np.array([.51, .48, .01])
    out_ = (np.array([[4, 3, 0], [5, 4, 1]]),
            np.array([ 0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.]))
    out = module.clamped_indexes(d, 3)
    np.testing.assert_allclose(out[0], out_[0])
    np.testing.assert_allclose(out[1], out_[1], rtol=1e-7, atol=1e-8)
