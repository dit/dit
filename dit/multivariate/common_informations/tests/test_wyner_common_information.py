"""
Tests for dit.multivariate.wyner_common_information
"""

from __future__ import division
import pytest

from dit import Distribution as D
from dit.multivariate import wyner_common_information as C
from dit.multivariate.common_informations.wyner_common_information import WynerCommonInformation
from dit.shannon import entropy


outcomes = ['0000', '0001', '0110', '0111', '1010', '1011', '1100', '1101']
pmf = [1/8]*8
xor = D(outcomes, pmf)

sbec = lambda p: D(['00', '0e', '1e', '11'], [(1-p)/2, p/2, p/2, (1-p)/2])
C_sbec = lambda p: 1 if p < 1/2 else entropy(p)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(('rvs', 'crvs', 'val'), [
    (None, None, 2.0),
    ([[0], [1], [2]], None, 2.0),
    ([[0], [1]], [2, 3], 1.0),
    ([[0], [1]], [2], 1.0),
    ([[0], [1]], None, 0.0),
])
def test_wci1(rvs, crvs, val):
    """
    Test against known values.
    """
    assert C(xor, rvs, crvs) == pytest.approx(val, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_wci2():
    """
    Test the golden mean.
    """
    gm = D([(0,0), (0,1), (1,0)], [1/3]*3)
    wci = WynerCommonInformation(gm, bound=2)
    wci.optimize()
    d = wci.construct_distribution(cutoff=1e-2)
    d_opt1 = D([(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0)], [1/6, 1/6, 1/3, 1/3])
    d_opt2 = D([(0, 0, 1), (0, 0, 0), (0, 1, 1), (1, 0, 0)], [1/6, 1/6, 1/3, 1/3])
    d_opt3 = D([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1)], [1/6, 1/6, 1/3, 1/3])
    d_opt4 = D([(0, 0, 1), (0, 0, 0), (0, 1, 0), (1, 0, 1)], [1/6, 1/6, 1/3, 1/3])
    d_opts = [d_opt1, d_opt2, d_opt3, d_opt4]
    equal = lambda d1, d2: d1.is_approx_equal(d2, rtol=1e-2, atol=1e-2)
    assert any(equal(d, d_opt) for d_opt in d_opts)
    assert wci.objective(wci._optima) == pytest.approx(2/3, abs=1e-3)


@pytest.mark.slow
@pytest.mark.skip("Computing Jacobians with numdifftools is too slow.")
def test_wci3():
    """
    Test with jacobian=True
    """
    pytest.importorskip("numdifftools")

    d = D([(0, 0), (1, 1)], [2/3, 1/3])
    wci = WynerCommonInformation(d, bound=2)
    wci._jacobian = True
    wci.optimize()
    d = wci.construct_distribution()
    d_opt1 = D([(0, 0, 0), (1, 1, 1)], [2/3, 1/3])
    d_opt2 = D([(0, 0, 1), (1, 1, 0)], [2/3, 1/3])
    d_opts = [d_opt1, d_opt2]
    assert any(d_opt.is_approx_equal(d, rtol=1e-4, atol=1e-4) for d_opt in d_opts)


@pytest.fixture
def x0():
    return {'x0': None}


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('i', range(1, 10))
def test_wci4(i, x0):
    """
    Test the binary symmetric erasure channel.
    """
    p = i/10
    wci = WynerCommonInformation(sbec(p))
    wci.optimize(x0=x0['x0'])
    x0['x0'] = wci._optima.copy()
    assert wci.objective(wci._optima) == pytest.approx(C_sbec(p), abs=1e-3)
