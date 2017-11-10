"""
Tests for the various common informations.
"""

import pytest

from hypothesis import given, settings, unlimited, HealthCheck

from dit.multivariate import (gk_common_information as K,
                              caekl_mutual_information as J,
                              dual_total_correlation as B,
                              wyner_common_information as C,
                              exact_common_information as G,
                              functional_common_information as F,
                              mss_common_information as M
                             )

from dit.utils.testing import distributions

pytest.importorskip('scipy')

epsilon = 1e-4

@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@settings(timeout=unlimited, min_satisfying_examples=3, max_examples=5, suppress_health_check=[HealthCheck.hung_test])
@given(dist=distributions(alphabets=(2,)*2))
def test_cis1(dist):
    """
    Test that the common informations are ordered correctly.
    """
    k = K(dist)
    j = J(dist)
    b = B(dist)
    c = C(dist)
    g = G(dist)
    f = F(dist)
    m = M(dist)
    assert k <= j + epsilon
    assert j <= b + epsilon
    assert b <= c + epsilon
    assert c <= g + epsilon
    assert g <= f + epsilon
    assert f <= m + epsilon

@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@settings(timeout=unlimited, min_satisfying_examples=3, max_examples=5, suppress_health_check=[HealthCheck.hung_test])
@given(dist=distributions(alphabets=(2,)*3))
def test_cis2(dist):
    """
    Test that the common informations are ordered correctly.
    """
    k = K(dist)
    j = J(dist)
    b = B(dist)
    c = C(dist)
    g = G(dist)
    f = F(dist)
    m = M(dist)
    assert k <= j + epsilon
    assert j <= b + epsilon
    assert b <= c + epsilon
    assert c <= g + epsilon
    assert g <= f + epsilon
    assert f <= m + epsilon
