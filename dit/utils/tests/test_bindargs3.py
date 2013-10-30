from dit.utils.bindargs import bindcallargs

from nose.tools import *

def F1(a, b, c=2, *, d, e, f=5, **kwargs):
    pass

def F2(*, d, e, f=5):
    pass

def test_bindcallargs1():
    out = ((0, 1, 2, 3, 4, 5), {'extra': 'hello'})
    assert_equal(bindcallargs(F1, 0, 1, d=3, e=4, extra='hello'), out)

def test_bindcallargs2():
    out = ((0, 1, 2), {})
    assert_equal(bindcallargs(F2, d=0, e=1, f=2), out)
    out = ((0, 1, 5), {})
    assert_equal(bindcallargs(F2, d=0, e=1), out)
    assert_raises(TypeError, bindcallargs, F2, d=0, f=2)
