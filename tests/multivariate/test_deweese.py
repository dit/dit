"""
Tests for dit.multivariate.deweese.
"""
import pytest

from dit.example_dists import dyadic, triadic, Xor
from dit.multivariate import (coinformation,
                              total_correlation,
                              dual_total_correlation,
                              caekl_mutual_information,
                              )
from dit.multivariate.deweese import (deweese_constructor,
                                      deweese_coinformation,
                                      deweese_total_correlation,
                                      deweese_dual_total_correlation,
                                      deweese_caekl_mutual_information,
                                      )


@pytest.mark.parametrize(('func', 'fast'), [
    (coinformation, deweese_coinformation),
    (total_correlation, deweese_total_correlation),
    (dual_total_correlation, deweese_dual_total_correlation),
    (caekl_mutual_information, deweese_caekl_mutual_information),
])
def test_constructor(func, fast):
    """ Test that constructor and opt give same answer """
    slow = deweese_constructor(func)
    val_1 = slow(Xor())
    val_2 = fast(Xor(), deterministic=True)
    assert val_1 == pytest.approx(val_2)


@pytest.mark.parametrize(('dist', 'true'), [
    (dyadic, 0.061278124458766986),
    (triadic, 1.0),
])
def test_deweese(dist, true):
    """ Test against known value """
    val = deweese_coinformation(dist)
    assert val == pytest.approx(true)


def test_return_dist():
    """ Test the return dist """
    f = deweese_constructor(coinformation)
    _, d = f(triadic, return_opt=True)
    d = d.marginal([4, 5, 6])
    assert len(d.outcomes) == 2
