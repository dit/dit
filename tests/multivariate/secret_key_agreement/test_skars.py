"""
Test the hierarchy of secret key agreement rates.
"""

import pytest

from hypothesis import given, settings

from dit.utils.testing import distributions
from tests._backends import backends

from dit.multivariate.secret_key_agreement import (lower_intrinsic_mutual_information,
                                                   secrecy_capacity_skar,
                                                   necessary_intrinsic_mutual_information,
                                                   minimal_intrinsic_mutual_information,
                                                   # reduced_intrinsic_mutual_information,
                                                   intrinsic_mutual_information,
                                                   upper_intrinsic_mutual_information)

eps = 1e-3


@pytest.mark.parametrize('backend', backends)
@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 3))
def test_hierarchy(dist, backend):
    """
    Test that the bounds are ordered correctly.
    """
    limi = lower_intrinsic_mutual_information(dist, [[0], [1]], [2])
    sc = secrecy_capacity_skar(dist, [[0], [1]], [2], backend=backend)
    nimi = necessary_intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    mimi = minimal_intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    # rimi = reduced_intrinsic_mutual_information(dist, [[0], [1]], [2])
    imi = intrinsic_mutual_information(dist, [[0], [1]], [2], backend=backend)
    uimi = upper_intrinsic_mutual_information(dist, [[0], [1]], [2])

    assert 0 <= limi + eps
    assert limi <= sc + eps
    assert sc <= nimi + eps
    assert nimi <= mimi + eps
    # assert mimi <= rimi + eps
    # assert rimi <= imi + eps
    assert mimi <= imi + eps
    assert imi <= uimi + eps
