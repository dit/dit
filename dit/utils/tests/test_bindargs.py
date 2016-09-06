"""
Tests for dit.utils.bindargs.
"""

from dit.utils.bindargs import bindcallargs

def F0(a, b=3, *args, **kwargs): # pylint: disable=unused-argument
    pass

def F1(a, b, c=2, *args, **kwargs): # pylint: disable=unused-argument
    pass

def F2(a, b=3): # pylint: disable=unused-argument
    pass

def test_bindcallargs0():
    out = bindcallargs(F0, 5, 4, 3, 2, 1, hello='there')
    out_ = (5, 4, 3, 2, 1), {'hello': 'there'}
    assert out == out_

def test_bindcallargs1():
    out = bindcallargs(F1, 0, 1, 3, 4, 5, extra='hello')
    out_ = ((0, 1, 3, 4, 5), {'extra': 'hello'})
    assert out == out_

def test_bindcallargs2():
    out = bindcallargs(F2, 0)
    out_ = ((0, 3), {})
    assert out == out_
