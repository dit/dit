"""
Tests for dit.example_dists.empirical.titanic
"""

import urllib.error

import pytest

from dit.example_dists.empirical import titanic
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H


@pytest.fixture(scope="module")
def d():
    """
    The Titanic distribution, fetched once per module. Skips if the source data
    is unreachable (e.g. no network).
    """
    try:
        return titanic()
    except (RuntimeError, urllib.error.URLError) as e:  # pragma: no cover
        pytest.skip(f"Titanic dataset unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The four random variables are named.
    """
    assert d.get_rv_names() == ("Class", "Sex", "Age", "Survived")


def test_support(d):
    """
    21 of the 24 possible joint outcomes occur in the 714 complete records.
    """
    assert len(d.outcomes) == 21


def test_entropy(d):
    """
    Test the joint entropy against its known value.
    """
    assert H(d) == pytest.approx(3.4971, abs=1e-4)


def test_survival_information(d):
    """
    Class, sex, and age jointly carry ~0.384 bits about survival.
    """
    cmi = I(d, [["Survived"], ["Class", "Sex", "Age"]])
    assert cmi == pytest.approx(0.3840, abs=1e-4)
