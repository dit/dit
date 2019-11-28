"""
Tests for dit.multivariate.secret_key_agreement.interactive_intrinsic_mutual_information
"""

import pytest

from dit.example_dists import n_mod_m
from dit.multivariate.secret_key_agreement import interactive_intrinsic_mutual_information


def test_iimi1():
    """
    Test against known value.
    """
    iimi = interactive_intrinsic_mutual_information(n_mod_m(3, 2), rvs=[[0], [1]], crvs=[2], rounds=1)
    assert iimi == pytest.approx(0.0)
