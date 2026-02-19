"""
Tests of dit.example_dists.miscellaneous
"""

import pytest

from dit.example_dists.miscellaneous import gk_pos_i_neg
from dit.multivariate import coinformation, gk_common_information


def test_gk_pos_i_neg1():
    """
    Test against known values.
    """
    gk = gk_common_information(gk_pos_i_neg)
    assert gk == pytest.approx(0.5435644431995964)


def test_gk_pos_i_neg2():
    """
    Test against known values.
    """
    i = coinformation(gk_pos_i_neg)
    assert i == pytest.approx(-0.33143555680040304)
