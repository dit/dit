"""
Tests for dit.divergences.hypercontractivity_coefficient.
"""

from __future__ import division

import pytest

from hypothesis import given, settings, unlimited, HealthCheck

from dit import Distribution
from dit.divergences import hypercontractivity_coefficient
from dit.exceptions import ditException
from dit.example_dists import dyadic, triadic
from dit.utils.testing import distributions


@pytest.mark.parametrize('rvs', [
    ([[0], [1]]),
    ([[0], [2]]),
    ([[1], [2]]),
    ([[0, 1], [2]]),
    ([[0, 2], [1]]),
    ([[1, 2], [0]]),
])
@pytest.mark.parametrize('dist', [dyadic, triadic])
def test_hypercontractivity_coefficient(dist, rvs):
    """ Test against known values """
    assert hypercontractivity_coefficient(dist, rvs) == pytest.approx(1.0)


def test_hypercontractivity_coefficient2():
    """
    Test against a known value.
    """
    d = Distribution(['00', '01', '10', '11'], [1/4]*4)
    hc = hypercontractivity_coefficient(d, [[0], [1]])
    assert hc == pytest.approx(0.0)


@pytest.mark.parametrize('rvs', [['X', 'Y', 'Z'], ['X']])
def test_hypercontractivity_coefficient_failure(rvs):
    """ Test that hypercontractivity_coefficient fails with len(rvs) != 2 """
    with pytest.raises(ditException):
        hypercontractivity_coefficient(dyadic, rvs)


@pytest.mark.slow
@given(dist1=distributions(alphabets=(2,)*2, nondegenerate=True),
       dist2=distributions(alphabets=(2,)*2, nondegenerate=True))
@settings(deadline=None,
          timeout=unlimited,
          min_satisfying_examples=3,
          max_examples=5,
          suppress_health_check=[HealthCheck.hung_test],
          )
def test_hypercontractivity_coefficient_tensorization(dist1, dist2):
    """
    Test tensorization:
        hc(X X' : Y Y') = max(hc(X:Y), hc(X', Y'))
    """
    import dit
    dit.ditParams['repr.print'] = True
    mixed = dist1.__matmul__(dist2)
    hc_mixed = hypercontractivity_coefficient(mixed, [[0, 2], [1, 3]])
    hc_a = hypercontractivity_coefficient(dist1, [[0], [1]])
    hc_b = hypercontractivity_coefficient(dist2, [[0], [1]])
    assert hc_mixed == pytest.approx(max(hc_a, hc_b), abs=1e-2)
