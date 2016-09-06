"""
Tests for dit.multivariate.exact_common_information
"""

from __future__ import division

import pytest

from dit import Distribution as D
from dit.multivariate import exact_common_information as G
from dit.multivariate.exact_common_information import ExactCommonInformation
from dit.shannon import entropy

outcomes = ['0000', '0001', '0110', '0111', '1010', '1011', '1100', '1101']
pmf = [1/8]*8
xor = D(outcomes, pmf)

sbec = lambda p: D(['00', '0e', '1e', '11'], [(1-p)/2, p/2, p/2, (1-p)/2])
G_sbec = lambda p: min(1, entropy(p) + 1 - p)


pytest.importorskip('scipy')

@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(('rvs', 'crvs', 'val'), [
    (None, None, 2.0),
    ([[0], [1], [2]], None, 2.0),
    ([[0], [1]], [2, 3], 1.0),
    ([[0], [1]], [2], 1.0),
    ([[0], [1]], None, 0.0),
])
def test_eci1(rvs, crvs, val):
    """
    Test against known values.
    """
    assert G(xor, rvs, crvs) == pytest.approx(val, abs=1e-4)

@pytest.fixture
def x0():
    return {'x0': None}

@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('i', range(1, 10))
def test_eci2(i, x0):
    """
    Test the binary symmetric erasure channel.
    """
    p = i/10
    eci = ExactCommonInformation(sbec(p))
    eci.optimize(x0=x0['x0'], nhops=5)
    x0['x0'] = eci._optima
    assert eci.objective(eci._optima) == pytest.approx(G_sbec(p), abs=1e-4)
