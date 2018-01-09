"""
Tests for dit.divergences.maximum_correlation.
"""

import pytest

from hypothesis import given

from dit.divergences import maximum_correlation
from dit.exceptions import ditException
from dit.example_dists import dyadic, triadic
from dit.utils.testing import distributions


@pytest.mark.parametrize(('rvs', 'crvs'), [
    ([[0], [1]], []),
    ([[0], [1]], [2]),
])
@pytest.mark.parametrize('dist', [dyadic, triadic])
def test_maximum_correlation(dist, rvs, crvs):
    """ Test against known values """
    assert maximum_correlation(dist, rvs, crvs) == pytest.approx(1.0)


@pytest.mark.parametrize('rvs', [['X', 'Y', 'Z'], ['X']])
def test_maximum_correlation_failure(rvs):
    """ Test that maximum_correlation fails with len(rvs) != 2 """
    with pytest.raises(ditException):
        maximum_correlation(dyadic, rvs)


@given(dist1=distributions(alphabets=((2, 4),)*2, nondegenerate=True),
       dist2=distributions(alphabets=((2, 4),)*2, nondegenerate=True))
def test_maximum_correlation_tensorization(dist1, dist2):
    """
    Test tensorization:
        rho(X X' : Y Y') = max(rho(X:Y), rho(X', Y'))
    """
    mixed = dist1.__matmul__(dist2)
    rho_mixed = maximum_correlation(mixed, [[0, 2], [1, 3]])
    rho_a = maximum_correlation(dist1, [[0], [1]])
    rho_b = maximum_correlation(dist2, [[0], [1]])
    assert rho_mixed == pytest.approx(max(rho_a, rho_b), abs=1e-4)
