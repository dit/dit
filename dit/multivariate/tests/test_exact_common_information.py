"""
Tests for dit.multivariate.exact_common_information
"""

from __future__ import division

from nose.plugins.attrib import attr
from nose.tools import assert_almost_equal

from dit import Distribution as D
from dit.multivariate import exact_common_information as G
from dit.multivariate.exact_common_information import ExactCommonInformation
from dit.shannon import entropy

outcomes = ['0000', '0001', '0110', '0111', '1010', '1011', '1100', '1101']
pmf = [1/8]*8
xor = D(outcomes, pmf)

sbec = lambda p: D(['00', '0e', '1e', '11'], [(1-p)/2, p/2, p/2, (1-p)/2])
G_sbec = lambda p: min(1, entropy(p) + 1 - p)

@attr('scipy')
@attr('slow')
def test_eci1():
    """
    Test against known values.
    """
    assert_almost_equal(G(xor), 2.0, places=4)
    assert_almost_equal(G(xor, [[0], [1], [2]]), 2.0, places=4)
    assert_almost_equal(G(xor, [[0], [1]], [2, 3]), 1.0, places=4)
    assert_almost_equal(G(xor, [[0], [1]], [2]), 1.0, places=4)
    assert_almost_equal(G(xor, [[0], [1]]), 0.0, places=4)

@attr('scipy')
@attr('slow')
def test_eci2():
    """
    Test the binary symmetric erasure channel.
    """
    x0 = None
    for p in [i/10 for i in range(1, 10)]:
        eci = ExactCommonInformation(sbec(p))
        eci.optimize(x0=x0, nhops=10)
        x0 = eci._res.x
        yield assert_almost_equal, eci.objective(eci._res.x), G_sbec(p), 4
