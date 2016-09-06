from dit.utils.bindargs import bindcallargs

import pytest

def F1(a, b, c=2, *, d, e, f=5, **kwargs):
    pass

def F2(*, d, e, f=5):
    pass

def F3(a, b, *args, c, d=7, e=8):
    pass

def test_bindcallargs1():
    out = bindcallargs(F1, 0, 1, d=3, e=4, extra='hello')
    out_ = ((0, 1, 2), {'d':3, 'e':4, 'f':5, 'extra': 'hello'})
    assert out == out_

def test_bindcallargs2():
    out = bindcallargs(F2, d=0, e=1, f=2)
    out_ = ((), {'d':0, 'e':1, 'f':2})
    assert out == out_
    out = bindcallargs(F2, d=0, e=1)
    out_ = ((), {'d':0, 'e':1, 'f':5})
    assert out == out_
    with pytest.raises(TypeError):
        bindcallargs(F2, d=0, f=2)

def test_bindcallargs3():
    out = bindcallargs(F3, 0, 1, 2, 3, 4, 5, c=6, e=-8)
    out_ = ((0, 1, 2, 3, 4, 5), {'c':6, 'd':7, 'e':-8})
    assert out == out_
