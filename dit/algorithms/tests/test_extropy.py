from __future__ import division

from nose.tools import assert_almost_equal, assert_true

import numpy as np

from dit import Distribution as D, ScalarDistribution as SD
from dit.algorithms import entropy as H, extropy as J

def test_J1():
    assert_almost_equal(H(0.25), J(0.25))

def test_J2():
    d = SD([1/2]*2)
    assert_almost_equal(J(d), 1)

def test_J3():
    d = D(['00', '11'], [1/2, 1/2])
    assert_almost_equal(J(d), 1)
    assert_almost_equal(J(d, [0,1]), 1)

# based on result 2 from the extropy paper
def test_J4():
    for i in range(2, 10):
        d = SD([1/i]*i)
        yield assert_almost_equal, J(d), (i-1)*(np.log2(i) - np.log2(i-1))

# based on result 2 from the extropy paper
def test_J5():
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base('e')
        yield assert_almost_equal, J(d), (i-1)*(np.log(i)-np.log(i-1))

# based on result 2 from the extropy paper
def test_J6():
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base(10)
        yield assert_almost_equal, J(d), (i-1)*(np.log10(i)-np.log10(i-1))

# nose on travisCI with python 2.6 doesn't have assert_less
# based on result 1 from the extropy paper
def test_J7():
    for i in range(3, 10):
        d = SD([1/i]*i)
        yield assert_true, J(d) < H(d)

# nose on travisCI with python 2.6 doesn't have assert_less
# based on result 1 from the extropy paper
def test_J8():
    for i in range(3, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        yield assert_true, J(d) < H(d)

# based on result 4 from extropy paper
def test_J9():
    d = SD([1/2, 1/4, 1/8, 1/16, 1/16])
    j = J(d)
    h = H(d)
    s = sum(H(p) for p in d.pmf)
    assert_almost_equal(j, s-h)

# based on result 4 from extropy paper
def test_J10():
    d = SD([1/2, 1/4, 1/8, 1/16, 1/16])
    j = J(d)
    h = H(d)
    s = sum(J(p) for p in d.pmf)
    assert_almost_equal(h, s-j)

# based on result 5 from extropy paper
def test_J11():
    pmf = np.array([1/2, 1/4, 1/8, 1/16, 1/16])
    N = len(pmf)
    d = SD(pmf)
    q = SD(1/(N-1)*(1-pmf))
    j2 = (N-1)*(H(q)-np.log2(N-1))
    assert_almost_equal(J(d), j2)
