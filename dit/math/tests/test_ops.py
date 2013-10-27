from __future__ import division

from nose.tools import *

import numpy as np
import numpy.testing as npt

from dit.exceptions import InvalidBase
from dit.math.ops import (
    get_ops, LinearOperations, LogOperations, exp_func, log_func
)

def test_get_ops():
    assert_true(isinstance(get_ops('linear'), LinearOperations))
    assert_true(isinstance(get_ops(2), LogOperations))

class TestLinear(object):
    def setUp(self):
        self.ops = LinearOperations()

    def test_add(self):
        X = np.array([0,1,0,-1,0,1,-1,1,-1])
        Y = np.array([0,0,1,0,-1,-1,1,1,-1])
        Z = np.array([0,1,1,-1,-1,0,0,2,-2])
        for x, y, z in zip(X, Y, Z):
            assert_almost_equal(self.ops.add(x, y), z)
        npt.assert_allclose(self.ops.add(X, Y), Z)

    def test_add_inplace(self):
        X = np.array([0,1,0,-1,0,1,-1,1,-1])
        Y = np.array([0,0,1,0,-1,-1,1,1,-1])
        Z = np.array([0,1,1,-1,-1,0,0,2,-2])
        self.ops.add_inplace(X, Y)
        npt.assert_allclose(X, Z)

    def test_add_reduce(self):
        X = np.array([[0,0,0],[0,1,2],[1,1,1],[-1,0,1],[2,0,-2],[-1,-1,-1]])
        Y = np.array([0,3,3,0,0,-3])
        for x, y in zip(X, Y):
            assert_almost_equal(self.ops.add_reduce(x), y)

    def test_mult(self):
        X = np.array([0,1,0,-1,0,1,-1,1,-1,2,2,2])
        Y = np.array([0,0,1,0,-1,-1,1,1,-1,1,2,-2])
        Z = np.array([0,0,0,0,0,-1,-1,1,1,2,4,-4])
        for x, y, z in zip(X, Y, Z):
            assert_almost_equal(self.ops.mult(x, y), z)
        npt.assert_allclose(self.ops.mult(X, Y), Z)

    def test_mult_inplace(self):
        X = np.array([0,1,0,-1,0,1,-1,1,-1,2,2,2])
        Y = np.array([0,0,1,0,-1,-1,1,1,-1,1,2,-2])
        Z = np.array([0,0,0,0,0,-1,-1,1,1,2,4,-4])
        self.ops.mult_inplace(X, Y)
        npt.assert_allclose(X, Z)

    def test_invert(self):
        X = np.array([1,2,-1,10], dtype=float)
        Y = np.array([1,1/2,-1,1/10])
        for x, y in zip(X, Y):
            assert_almost_equal(self.ops.invert(x), y)

    def test_mult_reduce(self):
        prods = [1, 2, 6, 24, 120]
        for i, p in enumerate(prods):
            assert_almost_equal(self.ops.mult_reduce(np.arange(1, i+2)), p)

class TestLog2(object):
    def setUp(self):
        self.ops = LogOperations(2)

    def test_add(self):
        X = self.ops.log(np.array([0,1,0,2,0,1,2,1,2]))
        Y = self.ops.log(np.array([0,0,1,0,2,2,1,1,2]))
        Z = self.ops.log(np.array([0,1,1,2,2,3,3,2,4]))
        for x, y, z in zip(X, Y, Z):
            npt.assert_allclose(self.ops.add(x, y), z)
        npt.assert_allclose(self.ops.add(X, Y), Z)

    def test_add_inplace(self):
        X = self.ops.log(np.array([0,1,0,2,0,1,2,1,2]))
        Y = self.ops.log(np.array([0,0,1,0,2,2,1,1,2]))
        Z = self.ops.log(np.array([0,1,1,2,2,3,3,2,4]))
        self.ops.add_inplace(X, Y)
        npt.assert_allclose(X, Z)

    def test_add_reduce(self):
        X = self.ops.log(np.array([[0,0,0],[0,1,2],[1,1,1]]))
        Y = self.ops.log(np.array([0,3,3]))
        for x, y in zip(X, Y):
            assert_almost_equal(self.ops.add_reduce(x), y)
        assert_almost_equal(self.ops.add_reduce(np.array([])), self.ops.zero)

    def test_mult(self):
        X = self.ops.log(np.array([0,1,0,.5,0,1,.5,1,.5,2,2,2]))
        Y = self.ops.log(np.array([0,0,1,0,5,.5,1,1,.5,1,2,.5]))
        Z = self.ops.log(np.array([0,0,0,0,0,.5,.5,1,.25,2,4,1]))
        for x, y, z in zip(X, Y, Z):
            assert_almost_equal(self.ops.mult(x, y), z)
        npt.assert_allclose(self.ops.mult(X, Y), Z)

    def test_mult_inplace(self):
        X = self.ops.log(np.array([0,1,0,.5,0,1,.5,1,.5,2,2,2]))
        Y = self.ops.log(np.array([0,0,1,0,5,.5,1,1,.5,1,2,.5]))
        Z = self.ops.log(np.array([0,0,0,0,0,.5,.5,1,.25,2,4,1]))
        self.ops.mult_inplace(X, Y)
        npt.assert_allclose(X, Z)

    def test_invert(self):
        X = self.ops.log(np.array([1,2,.5,10], dtype=float))
        Y = self.ops.log(np.array([1,1/2,2,1/10]))
        for x, y in zip(X, Y):
            assert_almost_equal(self.ops.invert(x), y)

    def test_mult_reduce(self):
        nums = np.arange(1,5+1)
        prods = np.cumprod(nums)
        prods = self.ops.log(prods)
        nums = self.ops.log(nums)
        print prods
        for i, p in enumerate(prods):
            assert_almost_equal(self.ops.mult_reduce(nums[:i+1]), p)
        assert_almost_equal(self.ops.mult_reduce(np.array([])), self.ops.one)

class TestLog3(TestLog2):
    def setUp(self):
        self.ops = LogOperations(3.5)

class TestLogE(TestLog2):
    def setUp(self):
        self.ops = LogOperations('e')

class TestLogHalf(TestLog2):
    def setUp(self):
        self.ops = LogOperations(.5)

def test_exp_func():
    bad_bases = ['pants', -1, 0, 1]
    for b in bad_bases:
        assert_raises(InvalidBase, exp_func, b)

def test_log_func():
    bad_bases = ['pants', -1, 0, 1]
    for b in bad_bases:
        assert_raises(InvalidBase, log_func, b)
