from __future__ import division

from nose.tools import *

from dit.math.misc import *

def test_number1():
    for i in range(-10, 10):
        assert_true(is_number(i))

def test_number2():
    for i in range(-10, 10):
        assert_true(is_number(i/10))

def test_number3():
    for i in ['a', int, []]:
        assert_false(is_number(i))

def test_integer1():
    for i in range(-10, 10):
        assert_true(is_integer(i))

def test_integer2():
    for i in range(-10, 10):
        assert_false(is_integer(i/10))

def test_integer3():
    for i in ['a', int, []]:
        assert_false(is_integer(i))

def test_factorial1():
    vals = [0, 1, 2, 3, 4, 5]
    facs = [1, 1, 2, 6, 24, 120]
    for v, f in zip(vals, facs):
        assert_equal(factorial(v), f)

def test_factorial2():
    vals = [-1, 0.5, 1+2j]
    for v in vals:
        assert_raises(ValueError, factorial, v)

def test_factorial3():
    vals = ['a', int, []]
    for v in vals:
        assert_raises(TypeError, factorial, v)

def test_combinations1():
    n = 3
    ks = [0, 1, 2, 3]
    cs = [1, 3, 3, 1]
    for k, c in zip(ks, cs):
        assert_equal(combinations(n, k), c)

def test_combinations2():
    assert_raises(ValueError, combinations, 5, 7)