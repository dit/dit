"""
Tests for the various common informations.
"""

import pytest
from hypothesis import given, settings

from dit.multivariate import (
    caekl_mutual_information as J,
)
from dit.multivariate import (
    dual_total_correlation as B,
)
from dit.multivariate import (
    exact_common_information as G,
)
from dit.multivariate import (
    functional_common_information as F,
)
from dit.multivariate import (
    gk_common_information as K,
)
from dit.multivariate import (
    mss_common_information as M,
)
from dit.multivariate import (
    wyner_common_information as C,
)
from dit.utils.testing import distributions
from tests._backends import backends

epsilon = 1e-2


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('backend', backends)
@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 2))
def test_cis1(dist, backend):
    """
    Test that the common informations are ordered correctly.
    """
    k = K(dist)
    j = J(dist)
    b = B(dist)
    c = C(dist, niter=100, backend=backend)
    g = G(dist, niter=100, backend=backend)
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
@pytest.mark.parametrize('backend', backends)
@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 3))
def test_cis2(dist, backend):
    """
    Test that the common informations are ordered correctly.
    """
    k = K(dist)
    j = J(dist)
    b = B(dist)
    c = C(dist, niter=100, backend=backend)
    g = G(dist, niter=100, backend=backend)
    f = F(dist)
    m = M(dist)
    assert k <= j + epsilon
    assert j <= b + epsilon
    assert b <= c + epsilon
    assert c <= g + epsilon
    assert g <= f + epsilon
    assert f <= m + epsilon
