"""
Tests for dit.multivariate.secret_key_agreement.reduced_intrinsic_mutual_information
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution, insert_rvf
from dit.example_dists.intrinsic import *
from dit.multivariate import (
    coinformation,
    entropy,
    intrinsic_total_correlation,
    reduced_intrinsic_CAEKL_mutual_information,
    reduced_intrinsic_dual_total_correlation,
    reduced_intrinsic_total_correlation,
    total_correlation,
)
from dit.multivariate.secret_key_agreement.reduced_intrinsic_mutual_informations import (
    ReducedIntrinsicTotalCorrelation,
)
from dit.utils.testing import distributions
from tests._backends import backends

# (dist, lower, upper). The reduced intrinsic MI upper bounds the secret key
# rate. For intrinsic_3 the optimum is not known in closed form; the upper value
# is a feasible point found by the optimizer and confirmed independently.
_dists = [
    (intrinsic_1, 0.0, 0.0),
    (intrinsic_2, 1.0, 1.0),
    (intrinsic_3, intrinsic_3.secret_rate, 0.99499),
]

_measures = [
    reduced_intrinsic_total_correlation,
    reduced_intrinsic_dual_total_correlation,
    reduced_intrinsic_CAEKL_mutual_information,
]


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("measure", _measures)
@pytest.mark.parametrize(("dist", "lower", "upper"), _dists)
def test_known_values(dist, lower, upper, measure, backend):
    """
    Test against known values.
    """
    rimi = measure(dist, [[0], [1]], [2], bounds=(2, 3, 4), backend=backend)
    assert lower - 1e-5 <= rimi <= upper + 1e-5


def test_objective_matches_definition():
    """
    The flattened objective equals I[X:Y|Zbar] + H[U] of the constructed joint,
    and Zbar depends on XY only through (Z, U).
    """
    rimi = ReducedIntrinsicTotalCorrelation(intrinsic_2, rvs=[[0], [1]], crvs=[2], bound=2)
    objective = rimi._objective()
    x = rimi.construct_random_initial()

    d = Distribution.from_ndarray(rimi.construct_joint(x))
    expected = coinformation(d, [[0], [1]], [4]) + entropy(d, [3])

    assert objective(rimi, x) == pytest.approx(expected)
    assert coinformation(d, [[0, 1], [4]], [2, 3]) == pytest.approx(0.0, abs=1e-10)


def test_fixed_u_reduces_to_intrinsic():
    """
    For a fixed U, the joint minimization over Zbar is the intrinsic MI
    conditioned on ZU, so the optimum is I[X:Y down ZU] + H[U].
    """
    d = insert_rvf(intrinsic_2, lambda o: ("1" if o[0] in "23" else "0",))
    value = intrinsic_total_correlation(d, [[0], [1]], [2, 3]) + entropy(d, [3])
    assert value == pytest.approx(1.0, abs=1e-5)
    assert reduced_intrinsic_total_correlation(intrinsic_2, [[0], [1]], [2], bounds=(2, 3, 4)) == pytest.approx(
        value, abs=1e-5
    )


@settings(max_examples=10)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_bounded_by_intrinsic(dist):
    """
    0 <= I[X:Y reduced Z] <= I[X:Y down Z] <= min(I[X:Y], I[X:Y|Z]).
    """
    rimi = reduced_intrinsic_total_correlation(dist, [[0], [1]], [2], bounds=(2,))
    imi = intrinsic_total_correlation(dist, [[0], [1]], [2])
    trivial = min(total_correlation(dist, [[0], [1]]), total_correlation(dist, [[0], [1]], [2]))
    assert -1e-9 <= rimi <= trivial + 1e-6
    assert rimi <= imi + 1e-4


def test_auxiliary_bounds():
    """
    U is bounded by the requested size and Zbar by |Z| * |U|.
    """
    rimi = ReducedIntrinsicTotalCorrelation(intrinsic_2, rvs=[[0], [1]], crvs=[2], bound=3)
    assert rimi._aux_bounds == [3, 2 * 3]
    assert np.prod(rimi.construct_joint(rimi.construct_random_initial()).shape) == 4 * 4 * 2 * 3 * 6
