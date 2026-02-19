"""
Tests for dit.multivariate.secret_key_agreement.interactive_intrinsic_mutual_information
"""

import pytest

from dit.example_dists import n_mod_m
from dit.multivariate.secret_key_agreement import interactive_intrinsic_mutual_information

from tests._backends import backends


@pytest.mark.parametrize('backend', backends)
def test_iimi1(backend):
    """
    Test against known value.
    """
    iimi = interactive_intrinsic_mutual_information(n_mod_m(3, 2), rvs=[[0], [1]], crvs=[2], rounds=1, backend=backend)
    assert iimi == pytest.approx(0.0)
