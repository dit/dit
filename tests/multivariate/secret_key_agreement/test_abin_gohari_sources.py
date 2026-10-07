"""
Tests for the Abin & Gohari example sources in dit.example_dists.intrinsic.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit.example_dists.intrinsic import deterministic_erasure, deterministic_erasure_source, xor_key
from dit.multivariate import entropy, total_correlation
from dit.multivariate.secret_key_agreement import (
    intrinsic_mutual_information,
    less_noisy_intrinsic_mutual_information,
    lower_intrinsic_mutual_information,
    minimal_intrinsic_mutual_information,
    one_way_skar,
    relaxed_two_part_intrinsic_mutual_information,
)
from dit.multivariate.secret_key_agreement.trivial_bounds import lower_intrinsic_mutual_information_directed


@settings(max_examples=25, deadline=None)
@given(eps=st.floats(min_value=0.0, max_value=1.0))
def test_deterministic_erasure_rate(eps):
    """
    I[X:Y|Z] = (1 - eps) H[X|Z], the claimed secret key agreement rate.
    """
    d = deterministic_erasure(eps)
    cmi = total_correlation(d, [[0], [1]], [2])
    assert cmi == pytest.approx((1 - eps) * entropy(d, [0], [2]), abs=1e-9)
    assert cmi == pytest.approx(d.secret_rate, abs=1e-9)


@pytest.mark.flaky(reruns=5)
def test_deterministic_erasure_one_way():
    """
    One-way communication from Alice achieves the rate.
    """
    d = deterministic_erasure_source
    assert one_way_skar(d, [0], [1], [2]) == pytest.approx(d.secret_rate, abs=1e-4)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(
    "measure",
    [
        intrinsic_mutual_information,
        relaxed_two_part_intrinsic_mutual_information,
    ],
)
def test_deterministic_erasure_upper_bounds(measure):
    """
    Upper bounds meet the one-way rate.
    """
    d = deterministic_erasure_source
    assert measure(d, [[0], [1]], [2]) == pytest.approx(d.secret_rate, abs=1e-4)


@settings(max_examples=25, deadline=None)
@given(eps=st.floats(min_value=0.0, max_value=1.0))
def test_deterministic_erasure_lower_bounds(eps):
    """
    I[X:Y] - I[Y:Z] equals the rate, while I[X:Y] - I[X:Z] = 1 - 2 eps falls short.
    """
    d = deterministic_erasure(eps)
    assert lower_intrinsic_mutual_information(d, [[0], [1]], [2]) == pytest.approx(d.secret_rate, abs=1e-9)
    assert lower_intrinsic_mutual_information_directed(d, [0], [1], [2]) == pytest.approx(
        max(0.0, 1 - 2 * eps), abs=1e-9
    )


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("pmf", [[0.4, 0.1, 0.2, 0.3], [0.1, 0.4, 0.4, 0.1], [0.25, 0.25, 0.1, 0.4]])
@pytest.mark.parametrize(
    "measure",
    [
        intrinsic_mutual_information,
        minimal_intrinsic_mutual_information,
        relaxed_two_part_intrinsic_mutual_information,
        less_noisy_intrinsic_mutual_information,
    ],
)
def test_xor_key_upper_bounds(pmf, measure):
    """
    With Z = X xor Y, every known upper bound equals I[X:Y].
    """
    d = xor_key(pmf)
    assert measure(d, [[0], [1]], [2]) == pytest.approx(total_correlation(d, [[0], [1]]), abs=1e-4)


def test_xor_key_rate_unknown():
    """
    The secret key agreement rate of the XOR key source is open.
    """
    assert not hasattr(xor_key([0.25] * 4), "secret_rate")
