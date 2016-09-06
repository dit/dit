"""
Tests for dit.math.ops.
"""

from __future__ import division

import pytest

import numpy as np

from dit.exceptions import InvalidBase
from dit.math.ops import (
    get_ops, LinearOperations, LogOperations, exp_func, log_func
)

def test_get_ops():
    assert isinstance(get_ops('linear'), LinearOperations)
    assert isinstance(get_ops(2), LogOperations)

class TestLinear(object):
    def setup_class(self):
        self.ops = LinearOperations()

    def test_add(self):
        X = np.array([0, 1, 0, -1, 0, 1, -1, 1, -1])
        Y = np.array([0, 0, 1, 0, -1, -1, 1, 1, -1])
        Z = np.array([0, 1, 1, -1, -1, 0, 0, 2, -2])
        for x, y, z in zip(X, Y, Z):
            assert self.ops.add(x, y) == pytest.approx(z)
        assert np.allclose(self.ops.add(X, Y), Z)

    def test_add_inplace(self):
        X = np.array([0, 1, 0, -1, 0, 1, -1, 1, -1])
        Y = np.array([0, 0, 1, 0, -1, -1, 1, 1, -1])
        Z = np.array([0, 1, 1, -1, -1, 0, 0, 2, -2])
        self.ops.add_inplace(X, Y)
        assert np.allclose(X, Z)

    def test_add_reduce(self):
        X = np.array([[0, 0, 0],
                      [0, 1, 2],
                      [1, 1, 1],
                      [-1, 0, 1],
                      [2, 0, -2],
                      [-1, -1, -1]])
        Y = np.array([0, 3, 3, 0, 0, -3])
        for x, y in zip(X, Y):
            assert self.ops.add_reduce(x) == pytest.approx(y)

    def test_mult(self):
        X = np.array([0, 1, 0, -1, 0, 1, -1, 1, -1, 2, 2, 2])
        Y = np.array([0, 0, 1, 0, -1, -1, 1, 1, -1, 1, 2, -2])
        Z = np.array([0, 0, 0, 0, 0, -1, -1, 1, 1, 2, 4, -4])
        for x, y, z in zip(X, Y, Z):
            assert self.ops.mult(x, y) == pytest.approx(z)
        assert np.allclose(self.ops.mult(X, Y), Z)

    def test_mult_inplace(self):
        X = np.array([0, 1, 0, -1, 0, 1, -1, 1, -1, 2, 2, 2])
        Y = np.array([0, 0, 1, 0, -1, -1, 1, 1, -1, 1, 2, -2])
        Z = np.array([0, 0, 0, 0, 0, -1, -1, 1, 1, 2, 4, -4])
        self.ops.mult_inplace(X, Y)
        assert np.allclose(X, Z)

    def test_invert(self):
        X = np.array([1, 2, -1, 10], dtype=float)
        Y = np.array([1, 1/2, -1, 1/10])
        for x, y in zip(X, Y):
            assert self.ops.invert(x) == pytest.approx(y)

    def test_mult_reduce(self):
        prods = [1, 2, 6, 24, 120]
        for i, p in enumerate(prods):
            assert self.ops.mult_reduce(np.arange(1, i+2)) == pytest.approx(p)

    def test_normalize(self):
        X = np.ones(3)
        Y = self.ops.normalize(X)
        Y_ = X / 3
        assert np.allclose(Y, Y_)


class TestLog2(object):
    def setup_class(self):
        self.ops = LogOperations(2)

    def test_add(self):
        X = self.ops.log(np.array([0, 1, 0, 2, 0, 1, 2, 1, 2]))
        Y = self.ops.log(np.array([0, 0, 1, 0, 2, 2, 1, 1, 2]))
        Z = self.ops.log(np.array([0, 1, 1, 2, 2, 3, 3, 2, 4]))
        for x, y, z in zip(X, Y, Z):
            assert np.allclose(self.ops.add(x, y), z)
        assert np.allclose(self.ops.add(X, Y), Z)

    def test_add_inplace(self):
        X = self.ops.log(np.array([0, 1, 0, 2, 0, 1, 2, 1, 2]))
        Y = self.ops.log(np.array([0, 0, 1, 0, 2, 2, 1, 1, 2]))
        Z = self.ops.log(np.array([0, 1, 1, 2, 2, 3, 3, 2, 4]))
        self.ops.add_inplace(X, Y)
        assert np.allclose(X, Z)

    def test_add_reduce(self):
        X = self.ops.log(np.array([[0, 0, 0], [0, 1, 2], [1, 1, 1]]))
        Y = self.ops.log(np.array([0, 3, 3]))
        for x, y in zip(X, Y):
            assert np.allclose(self.ops.add_reduce(x), y)
        assert np.allclose(self.ops.add_reduce(np.array([])), self.ops.zero)

    def test_mult(self):
        X = self.ops.log(np.array([0, 1, 0, 0.5, 0, 1, 0.5, 1, 0.5, 2, 2, 2]))
        Y = self.ops.log(np.array([0, 0, 1, 0, 5, 0.5, 1, 1, 0.5, 1, 2, 0.5]))
        Z = self.ops.log(np.array([0, 0, 0, 0, 0, 0.5, 0.5, 1, 0.25, 2, 4, 1]))
        for x, y, z in zip(X, Y, Z):
            assert np.allclose(self.ops.mult(x, y), z)
        assert np.allclose(self.ops.mult(X, Y), Z)

    def test_mult_inplace(self):
        X = self.ops.log(np.array([0, 1, 0, 0.5, 0, 1, 0.5, 1, 0.5, 2, 2, 2]))
        Y = self.ops.log(np.array([0, 0, 1, 0, 5, 0.5, 1, 1, 0.5, 1, 2, 0.5]))
        Z = self.ops.log(np.array([0, 0, 0, 0, 0, 0.5, 0.5, 1, 0.25, 2, 4, 1]))
        self.ops.mult_inplace(X, Y)
        assert np.allclose(X, Z)

    def test_invert(self):
        X = self.ops.log(np.array([1, 2, 0.5, 10], dtype=float))
        Y = self.ops.log(np.array([1, 1/2, 2, 1/10]))
        for x, y in zip(X, Y):
            assert np.allclose(self.ops.invert(x), y)

    def test_mult_reduce(self):
        nums = np.arange(1, 5+1)
        prods = np.cumprod(nums)
        prods = self.ops.log(prods)
        nums = self.ops.log(nums)
        for i, p in enumerate(prods):
            assert np.allclose(self.ops.mult_reduce(nums[:i+1]), p)
        assert np.allclose(self.ops.mult_reduce(np.array([])), self.ops.one)

    def test_normalize(self):
        W = np.ones(3)
        X = self.ops.log(W)
        Y = self.ops.normalize(X)
        Z = self.ops.exp(Y)
        Z_ = W / 3
        assert np.allclose(Z, Z_)


class TestLog3(TestLog2):
    def setup_class(self):
        self.ops = LogOperations(3.5)

class TestLogE(TestLog2):
    def setup_class(self):
        self.ops = LogOperations('e')

class TestLogHalf(TestLog2):
    def setup_class(self):
        self.ops = LogOperations(0.5)

@pytest.mark.parametrize('base', ['pants', -1, 0, 1])
def test_exp_func1(base):
    with pytest.raises(InvalidBase):
        exp_func(base)

def test_exp_func2():
    ops = LogOperations(0.5)
    assert np.allclose([0.5**1, 0.5**2, 0.5**3], ops.exp([1, 2, 3]))

@pytest.mark.parametrize('base', ['pants', -1, 0, 1])
def test_log_func1(base):
    with pytest.raises(InvalidBase):
        log_func(base)

def test_log_func2():
    ops = LogOperations(2)
    assert np.allclose([np.log2(1), np.log2(2), np.log2(3)], ops.log([1, 2, 3]))
