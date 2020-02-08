"""
Tests for the MDBSI distributions.
"""

import pytest

from dit.example_dists.mdbsi import dyadic, triadic
from dit.multivariate import (
    entropy as H,
    total_correlation as T,
    dual_total_correlation as B,
    coinformation as I,
    residual_entropy as R,
    caekl_mutual_information as J,
    tse_complexity as TSE,
    gk_common_information as K,
    wyner_common_information as C,
    exact_common_information as G,
    functional_common_information as F,
    mss_common_information as M,
)
from dit.other import (
    extropy as X,
    perplexity as P,
    disequilibrium as D,
)


@pytest.mark.parametrize(('measure', 'dy_val', 'tri_val'), [
    (H, 3, 3),
    (T, 3, 3),
    (B, 3, 3),
    (I, 0, 0),
    (R, 0, 0),
    (J, 1.5, 1.5),
    (TSE, 2, 2),
    (K, 0, 1),
    (C, 3, 3),
    (G, 3, 3),
    (F, 3, 3),
    (M, 3, 3),
    (X, 1.3485155455967712, 1.3485155455967712),
    (P, 8, 8),
    (D, 0.76124740551164605, 0.76124740551164605),
])
def test_measures(measure, dy_val, tri_val):
    """
    Test that the distributions have the correct properties.
    """
    assert measure(dyadic) == pytest.approx(dy_val)
    assert measure(triadic) == pytest.approx(tri_val)
