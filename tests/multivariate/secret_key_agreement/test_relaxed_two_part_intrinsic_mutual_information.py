"""
Tests for dit.multivariate.secret_key_agreement.relaxed_two_part_intrinsic_mutual_informations.
"""

import pytest

from dit.example_dists.intrinsic import *
from dit.exceptions import ditException
from dit.multivariate.secret_key_agreement.relaxed_two_part_intrinsic_mutual_informations import (
    RelaxedTwoPartIntrinsicMutualInformation,
    relaxed_two_part_intrinsic_mutual_information,
)
from tests._backends import backends


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("dist", [intrinsic_1, intrinsic_2, intrinsic_3])
def test_known_values(dist, backend):
    """
    Test against known values.
    """
    value = relaxed_two_part_intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    assert value == pytest.approx(dist.secret_rate, abs=1e-3)


def test_constant_initial_is_mutual_information():
    """
    With constant J and copy Zbar, the objective is I[X:Y].
    """
    opt = RelaxedTwoPartIntrinsicMutualInformation(intrinsic_1, [[0], [1]], [2], bound_j=2)
    opt.optimize()
    assert opt.objective(opt.construct_constant_initial()) == pytest.approx(1.5)


def test_requires_crvs():
    """
    Test that a conditional variable is required.
    """
    with pytest.raises(ditException):
        RelaxedTwoPartIntrinsicMutualInformation(intrinsic_1, [[0], [1]], [])
