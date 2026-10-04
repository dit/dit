"""
Test the hierarchy of secret key agreement rates.
"""

import pytest
from hypothesis import given, settings

from dit.distconst import uniform
from dit.multivariate.secret_key_agreement import (
    # reduced_intrinsic_mutual_information,
    intrinsic_mutual_information,
    lower_intrinsic_mutual_information,
    minimal_intrinsic_mutual_information,
    necessary_intrinsic_mutual_information,
    no_communication_skar,
    relaxed_two_part_intrinsic_mutual_information,
    secrecy_capacity_skar,
    upper_intrinsic_mutual_information,
)
from dit.utils.testing import distributions
from tests._backends import backends

eps = 1e-3


@pytest.mark.parametrize("backend", backends)
@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 3))
def test_hierarchy(dist, backend):
    """
    Test that the bounds are ordered correctly.
    """
    ncskar = no_communication_skar(dist, [0], [1], [2])
    limi = lower_intrinsic_mutual_information(dist, [[0], [1]], [2])
    sc = secrecy_capacity_skar(dist, [[0], [1]], [2], backend=backend)
    nimi = necessary_intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    mimi = minimal_intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    rtpimi = relaxed_two_part_intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    # rimi = reduced_intrinsic_mutual_information(dist, [[0], [1]], [2])
    imi = intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    uimi = upper_intrinsic_mutual_information(dist, [[0], [1]], [2])

    assert ncskar + eps >= 0
    assert limi + eps >= 0
    assert ncskar <= sc + eps
    assert limi <= sc + eps
    assert sc <= nimi + eps
    assert nimi <= rtpimi + eps
    assert rtpimi <= mimi + eps
    # assert mimi <= rimi + eps
    # assert rimi <= imi + eps
    assert mimi <= imi + eps
    assert imi <= uimi + eps


def test_no_communication_exceeds_lower_intrinsic():
    """
    The no-communication rate and the lower intrinsic mutual information are
    incomparable: here X = (W, A), Y = (W, B), Z = (A, B) for independent bits
    W, A, B, so Alice and Bob share W secretly but each pairwise mutual
    information is 1 bit.
    """
    outcomes = [(w + a, w + b, a + b) for w in "01" for a in "01" for b in "01"]
    dist = uniform(outcomes)
    assert no_communication_skar(dist, [0], [1], [2]) == pytest.approx(1)
    assert lower_intrinsic_mutual_information(dist, [[0], [1]], [2]) == pytest.approx(0)
    assert secrecy_capacity_skar(dist, [[0], [1]], [2]) == pytest.approx(1, abs=1e-4)
