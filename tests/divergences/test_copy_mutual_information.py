"""
Tests for dit.divergences.copy_mutual_information
"""

import pytest

from dit import Distribution
from dit.divergences.copy_mutual_information import copy_mutual_information


@pytest.mark.parametrize(
    ["dist", "value"],
    [
        (Distribution(["00", "11"], [1 / 2] * 2), 1.0),
        (Distribution(["01", "10"], [1 / 2] * 2), 0.0),
        (Distribution(["00", "12", "21", "33"], [1 / 4] * 4), 1.0),
    ],
)
def test_cmi_1(dist, value):
    """
    Test that equivalent distributions have zero metric.
    """
    cmi = copy_mutual_information(dist, [0], [1])
    assert cmi == pytest.approx(value)


def test_cmi_value_absent_from_y():
    """
    X may take values that Y never takes.
    """
    dist = Distribution(["00", "11", "20"], [1 / 3] * 3)
    assert copy_mutual_information(dist, [0], [1]) >= 0
