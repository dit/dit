"""
Tests for dit.multivariate.tse_complexity.
"""

from __future__ import division

import pytest

from dit import Distribution as D
from dit.multivariate import binding_information as B, tse_complexity as TSE
from dit.example_dists import n_mod_m
from dit.math.misc import combinations as nCk
from dit.utils import powerset

@pytest.mark.parametrize(('i', 'j'), list(zip(range(3, 6), range(2, 5))))
def test_tse1(i, j):
    """ Test identity comparing TSE to B from Olbrich's talk """
    d = n_mod_m(i, j)
    indices = [[k] for k in range(i)]
    tse = TSE(d)
    x = 1/2 * sum(B(d, rv)/nCk(i, len(rv)) for rv in powerset(indices))
    assert tse == pytest.approx(x)

@pytest.mark.parametrize('n', range(2, 7))
def test_tse2(n):
    """ Test TSE for giant bit distributions """
    d = D(['0'*n, '1'*n], [1/2, 1/2])
    tse = TSE(d)
    assert tse == pytest.approx((n-1)/2)
