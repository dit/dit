"""
Test the hierarchy of secret key agreement rates.
"""

from hypothesis import given, settings, unlimited, HealthCheck

from dit.utils.testing import distributions

from dit.multivariate.secret_key_agreement import (lower_intrinsic_mutual_information,
                                                   secrecy_capacity,
                                                   necessary_intrinsic_mutual_information,
                                                   minimal_intrinsic_mutual_information,
                                                   # reduced_intrinsic_mutual_information,
                                                   intrinsic_mutual_information,
                                                   upper_intrinsic_mutual_information)

eps = 1e-4


@settings(deadline=None,
          timeout=unlimited,
          min_satisfying_examples=3,
          max_examples=5,
          suppress_health_check=[HealthCheck.hung_test])
@given(dist=distributions(alphabets=(2,)*3))
def test_hierarchy(dist):
    """
    Test that the bounds are ordered correctly.
    """
    limi = lower_intrinsic_mutual_information(dist, [[0], [1]], [2])
    sc = secrecy_capacity(dist, [[0], [1]], [2])
    nimi = necessary_intrinsic_mutual_information(dist, [[0], [1]], [2])
    mimi = minimal_intrinsic_mutual_information(dist, [[0], [1]], [2])
    # rimi = reduced_intrinsic_mutual_information(dist, [[0], [1]], [2])
    imi = intrinsic_mutual_information(dist, [[0], [1]], [2])
    uimi = upper_intrinsic_mutual_information(dist, [[0], [1]], [2])

    assert 0 <= limi + eps
    assert limi <= sc + eps
    assert sc <= nimi + eps
    assert nimi <= mimi + eps
    # assert mimi <= rimi + eps
    # assert rimi <= imi + eps
    assert mimi <= imi + eps
    assert imi <= uimi + eps
