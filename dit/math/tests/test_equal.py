"""
Tests for dit.math.equal.
"""

from __future__ import division

from nose.tools import assert_false, assert_true

from numpy import inf, nan

from dit.math.equal import allclose, close__python as pyclose, close

def test_close1():
    x = 0
    y = 1
    assert_false(close(x, y))
    assert_false(pyclose(x, y))

def test_close2():
    x = 0
    y = inf
    assert_false(close(x, y))
    assert_false(pyclose(x, y))

def test_close3():
    x = -inf
    y = inf
    assert_false(close(x, y))
    assert_false(pyclose(x, y))

def test_close4():
    x = inf
    y = nan
    assert_false(close(x, y))
    assert_false(pyclose(x, y))

def test_close5():
    x = 0
    y = nan
    assert_false(close(x, y))
    assert_false(pyclose(x, y))

def test_close6():
    x = 1
    y = 1
    assert_true(close(x, y))
    assert_true(pyclose(x, y))

def test_close7():
    x = 0.33333333333333
    y = 1/3
    assert_true(close(x, y))
    assert_true(pyclose(x, y))

def test_close8():
    x = inf
    y = inf
    assert_true(close(x, y))
    assert_true(pyclose(x, y))

def test_allclose1():
    x = [0, 0, -inf, inf, 0]
    y = [1, inf, inf, nan, nan]
    assert_false(allclose(x, y))

def test_allclose2():
    x = [1, 0.333333333333333, inf]
    y = [1, 1/3, inf]
    assert_true(allclose(x, y))
