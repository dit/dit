from nose.tools import *

from dit.utils.misc import flatten, is_string_like, partitions, partitions2, \
                           ordered_partitions
from six import u

def test_flatten1():
    x = [[[0], 1, 2], 3, [4]]
    fx = list(flatten(x))
    y = [0, 1, 2, 3, 4]
    assert_equal(len(fx), len(y))
    for i, j in zip(fx, y):
        assert_equal(i, j)

def test_is_string_like1():
    ys = ['', 'hi', "pants", '"test"', u('pants'), r'pants']
    ns = [1, [], int, {}, ()]
    for y in ys:
        assert_true(is_string_like(y))
    for n in ns:
        assert_false(is_string_like(n))

def test_partitions1():
    bells = [1, 2, 5, 15, 52]
    for i, b in enumerate(bells):
        parts = list(partitions(range(i+1)))
        assert_equal(len(parts), b)

def test_ordered_partitions1():
    obells = [1, 1, 3, 13, 75]
    for i, b in enumerate(obells):
        oparts = list(ordered_partitions(range(i)))
        assert_equal(len(oparts), b)