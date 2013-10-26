from __future__ import division

from nose.tools import *

import numpy as np

from dit.exceptions import InvalidBase
from dit.math.ops import LinearOperations, exp_func, log_func

def test_linear_add():
    ops = LinearOperations()
    assert_almost_equal(ops.add(0, 0), 0)
    assert_almost_equal(ops.add(1, 0), 1)
    assert_almost_equal(ops.add(0, 1), 1)
    assert_almost_equal(ops.add(-1, 0), -1)
    assert_almost_equal(ops.add(0, -1), -1)
    assert_almost_equal(ops.add(1, -1), 0)
    assert_almost_equal(ops.add(-1, 1), 0)
    assert_almost_equal(ops.add(1, 1), 2)
    assert_almost_equal(ops.add(-1, -1), -2)

def test_linear_add_reduce():
    ops = LinearOperations()
    assert_almost_equal(ops.add_reduce(np.array([0,0,0])), 0)
    assert_almost_equal(ops.add_reduce(np.array([0,1,2])), 3)
    assert_almost_equal(ops.add_reduce(np.array([1,1,1])), 3)
    assert_almost_equal(ops.add_reduce(np.array([-1,0,1])), 0)
    assert_almost_equal(ops.add_reduce(np.array([2,0,-2])), 0)
    assert_almost_equal(ops.add_reduce(np.array([-1,-1,-1])), -3)

def test_linear_mult():
    ops = LinearOperations()
    assert_almost_equal(ops.mult(0, 0), 0)
    assert_almost_equal(ops.mult(1, 0), 0)
    assert_almost_equal(ops.mult(0, 1), 0)
    assert_almost_equal(ops.mult(-1, 0), 0)
    assert_almost_equal(ops.mult(0, -1), 0)
    assert_almost_equal(ops.mult(1, -1), -1)
    assert_almost_equal(ops.mult(-1, 1), -1)
    assert_almost_equal(ops.mult(1, 1), 1)
    assert_almost_equal(ops.mult(-1, -1), 1)
    assert_almost_equal(ops.mult(2, 1), 2)
    assert_almost_equal(ops.mult(2, 2), 4)
    assert_almost_equal(ops.mult(2, -2), -4)

def test_linear_invert():
    ops = LinearOperations()
    assert_almost_equal(ops.invert(1), 1)
    assert_almost_equal(ops.invert(2.0), 1/2)
    assert_almost_equal(ops.invert(-1), -1)
    assert_almost_equal(ops.invert(10.0), 1/10)

def test_linear_mult_reduce():
    ops = LinearOperations()
    prods = [1, 2, 6, 24, 120]
    for i, p in enumerate(prods):
        assert_equal(ops.mult_reduce(np.arange(1, i+2)), p)

def test_exp_func():
    bad_bases = ['pants', -1, 0, 1]
    for b in bad_bases:
        assert_raises(InvalidBase, exp_func, b)

def test_log_func():
    bad_bases = ['pants', -1, 0, 1]
    for b in bad_bases:
        assert_raises(InvalidBase, log_func, b)
