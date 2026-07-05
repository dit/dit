"""
Tests for dit.example_dists.empirical.blood_types
"""

import pytest

from dit.example_dists.empirical import blood_types
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H
from dit.multivariate import total_correlation as T


@pytest.fixture(scope="module")
def d():
    """
    The blood types distribution, built once per module.
    """
    return blood_types()


def test_normalized(d):
    """
    The pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The seven random variables are named.
    """
    assert d.get_rv_names() == (
        "Region", "ABO", "Rh", "Kell", "Duffy", "Kidd", "MNS"
    )


def test_support(d):
    """
    The joint has 5 * 4 * 2 * 2 * 4 * 3 * 9 = 8640 outcomes.
    """
    assert len(d.outcomes) == 8640


def test_entropy(d):
    """
    Test the joint entropy against its known value.
    """
    assert H(d) == pytest.approx(8.6925, abs=1e-4)


def test_geographic_information(d):
    """
    The antigen systems jointly carry ~0.699 bits about geography, dominated by
    Duffy.
    """
    systems = ["ABO", "Rh", "Kell", "Duffy", "Kidd", "MNS"]
    assert I(d, [["Region"], systems]) == pytest.approx(0.6991, abs=1e-4)
    assert I(d, [["Region"], ["Duffy"]]) == pytest.approx(0.5689, abs=1e-4)


def test_conditional_independence(d):
    """
    The antigen systems are conditionally independent given Region: both the
    conditional total correlation and any pairwise conditional mutual
    information vanish.
    """
    systems = [["ABO"], ["Rh"], ["Kell"], ["Duffy"], ["Kidd"], ["MNS"]]
    assert T(d, systems, ["Region"]) == pytest.approx(0.0, abs=1e-9)
    assert I(d, [["Duffy"], ["Kidd"]], ["Region"]) == pytest.approx(0.0, abs=1e-9)
