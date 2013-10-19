from nose.tools import *

from dit.utils.misc import flatten, is_string_like

def test_flatten1():
    x = [[[0], 1, 2], 3, [4]]
    fx = list(flatten(x))
    y = [0, 1, 2, 3, 4]
    assert_equal(len(fx), len(y))
    for i, j in zip(fx, y):
        assert_equal(i, j)

def test_is_string_like1():
    ys = ['', 'hi', "pants", '"test"', unicode('pants'), r'pants']
    ns = [1, [], int, {}, ()]
    for y in ys:
        assert_true(is_string_like(y))
    for n in ns:
        assert_false(is_string_like(n))