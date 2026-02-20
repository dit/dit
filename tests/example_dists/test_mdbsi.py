"""
Tests for the MDBSI distributions.
"""

import pytest

from dit.example_dists.mdbsi import dyadic, triadic
from dit.multivariate import (
    caekl_mutual_information as J,
)
from dit.multivariate import (
    coinformation as I,
)
from dit.multivariate import (
    dual_total_correlation as B,
)
from dit.multivariate import (
    entropy as H,
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
    residual_entropy as R,
)
from dit.multivariate import (
    total_correlation as T,
)
from dit.multivariate import (
    tse_complexity as TSE,
)
from dit.multivariate import (
    wyner_common_information as C,
)
from dit.other import (
    disequilibrium as D,
)
from dit.other import (
    extropy as X,
)
from dit.other import (
    perplexity as P,
)


@pytest.mark.parametrize(
    ("measure", "dy_val", "tri_val"),
    [
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
    ],
)
def test_measures(measure, dy_val, tri_val):
    """
    Test that the distributions have the correct properties.
    """
    assert measure(dyadic) == pytest.approx(dy_val)
    assert measure(triadic) == pytest.approx(tri_val)
