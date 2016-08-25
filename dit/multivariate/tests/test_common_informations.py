"""
Tests for the various common informations.
"""

import pytest

from dit import random_distribution
from dit.multivariate import (gk_common_information as K,
                              dual_total_correlation as B,
                              wyner_common_information as C,
                              exact_common_information as G,
                              functional_common_information as F,
                              mss_common_information as M
                             )


pytest.importorskip('scipy')

@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('d', [random_distribution(2, 3) for _ in range(5)])
def test_cis1(d):
    """
    Test that the common informations are ordered correctly.
    """
    k = K(d)
    b = B(d)
    c = C(d)
    g = G(d)
    f = F(d)
    m = M(d)
    assert k <= b + 1e-6
    assert b <= c + 1e-6
    assert c <= g + 1e-6
    assert g <= f + 1e-6
    assert f <= m + 1e-6

@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize('d', [random_distribution(3, 2) for _ in range(5)])
def test_cis2(d):
    """
    Test that the common informations are ordered correctly.
    """
    k = K(d)
    b = B(d)
    c = C(d)
    g = G(d)
    f = F(d)
    m = M(d)
    assert k <= b + 1e-6
    assert b <= c + 1e-6
    assert c <= g + 1e-6
    assert g <= f + 1e-6
    assert f <= m + 1e-6
