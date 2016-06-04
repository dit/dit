"""
Tests for dit.multivariate.wyner_common_information
"""

from __future__ import division

from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
from nose.tools import assert_almost_equal, assert_true

from dit import Distribution as D
from dit.multivariate import wyner_common_information as C
from dit.multivariate.wyner_common_information import WynerCommonInformation
from dit.shannon import entropy

outcomes = ['0000', '0001', '0110', '0111', '1010', '1011', '1100', '1101']
pmf = [1/8]*8
xor = D(outcomes, pmf)

sbec = lambda p: D(['00', '0e', '1e', '11'], [(1-p)/2, p/2, p/2, (1-p)/2])
C_sbec = lambda p: 1 if p < 1/2 else entropy(p)

@attr('scipy')
@attr('slow')
def test_wci1():
    """
    Test against known values.
    """
    assert_almost_equal(C(xor), 2.0, places=4)
    assert_almost_equal(C(xor, [[0], [1], [2]]), 2.0, places=4)
    assert_almost_equal(C(xor, [[0], [1]], [2, 3]), 1.0, places=4)
    assert_almost_equal(C(xor, [[0], [1]], [2]), 1.0, places=4)
    assert_almost_equal(C(xor, [[0], [1]]), 0.0, places=4)

@attr('scipy')
def test_wci2():
    """
    Test the golden mean.
    """
    gm = D([(0,0), (0,1), (1,0)], [1/3]*3)
    wci = WynerCommonInformation(gm)
    wci.optimize(minimize=True)
    d = wci.construct_distribution()
    d_opt1 = D([(0,0,0), (0,0,1), (0,1,1), (1,0,0)], [1/6, 1/6, 1/3, 1/3])
    d_opt2 = D([(0,0,1), (0,0,0), (0,1,1), (1,0,0)], [1/6, 1/6, 1/3, 1/3])
    d_opt3 = D([(0,0,0), (0,0,1), (0,1,0), (1,0,1)], [1/6, 1/6, 1/3, 1/3])
    d_opt4 = D([(0,0,1), (0,0,0), (0,1,0), (1,0,1)], [1/6, 1/6, 1/3, 1/3])
    d_opts = [d_opt1, d_opt2, d_opt3, d_opt4]
    equal = lambda d1, d2: d1.is_approx_equal(d2, rtol=1e-4, atol=1e-4)
    assert_true(any(equal(d, d_opt) for d_opt in d_opts))
    assert_almost_equal(wci.objective(wci._res.x), 2/3, places=5)

@attr('scipy')
@attr('slow')
def test_wci3():
    """
    Test with jacobian=True
    """
    # jacobian doesn't seem to work well right now
    raise SkipTest
    d = D([(0,0), (1,1)], [2/3, 1/3])
    wci = WynerCommonInformation(d)
    wci.optimize(minimize=True, jacobian=True)
    d_opt = D([(0,0,0), (1,1,1)], [2/3, 1/3])
    assert_true(d_opt.is_approx_equal(wci.construct_distribution()))

@attr('scipy')
@attr('slow')
def test_wci4():
    """
    Test the binary symmetric erasure channel.
    """
    for p in [i/10 for i in range(1, 10)]:
        yield assert_almost_equal, C(sbec(p)), C_sbec(p), 4
