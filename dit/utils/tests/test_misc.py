from nose.tools import *

from dit.utils.misc import flatten, is_string_like

def test_flatten1():
    x = [[[0], 1, 2], 3, [4]]
    y = [0, 1, 2, 3, 4]
    assert_list_equal(list(flatten(x)), y)

def test_is_string_like1():
    ys = ['', 'hi', "pants", '"test"', u'pants', r'pants']
    ns = [1, [], int, {}, ()]
    for y in ys:
        assert_true(is_string_like(y))
    for n in ns:
        assert_false(is_string_like(n))